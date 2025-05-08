# bot.py
import discord
from discord.ext import commands
import os
import json
import logging
import re
import requests
from report import Report
import pdb
from discord.ui import Button, View

# Set up logging to the console
logger = logging.getLogger('discord')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(handler)

# There should be a file called 'tokens.json' inside the same folder as this file
token_path = 'tokens.json'
if not os.path.isfile(token_path):
    raise Exception(f"{token_path} not found!")
with open(token_path) as f:
    # If you get an error here, it means your token is formatted incorrectly. Did you put it in quotes?
    tokens = json.load(f)
    discord_token = tokens['discord']


class ModBot(discord.Client):
    def __init__(self): 
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True  # Need this for blocking users
        super().__init__(intents=intents)
        self.group_num = None
        self.mod_channels = {} # Map from guild to the mod channel id for that guild
        self.reports = {} # Map from user IDs to the state of their report

    async def on_ready(self):
        print(f'{self.user.name} has connected to Discord! It is these guilds:')
        for guild in self.guilds:
            print(f' - {guild.name}')
        print('Press Ctrl-C to quit.')

        # Parse the group number out of the bot's name
        match = re.search('[gG]roup (\d+) [bB]ot', self.user.name)
        if match:
            self.group_num = match.group(1)
        else:
            raise Exception("Group number not found in bot's name. Name format should be \"Group # Bot\".")

        # Find the mod channel in each guild that this bot should report to
        for guild in self.guilds:
            for channel in guild.text_channels:
                if channel.name == f'group-{self.group_num}-mod':
                    self.mod_channels[guild.id] = channel
        

    async def on_message(self, message):
        '''
        This function is called whenever a message is sent in a channel that the bot can see (including DMs). 
        Currently the bot is configured to only handle messages that are sent over DMs or in your group's "group-#" channel. 
        '''
        # Ignore messages from the bot 
        if message.author.id == self.user.id:
            return

        # Check if this message was sent in a server ("guild") or if it's a DM
        if message.guild:
            await self.handle_channel_message(message)
        else:
            await self.handle_dm(message)

    async def on_interaction(self, interaction: discord.Interaction):
        custom_id = interaction.data.get("custom_id", "")
        mod_channel = self.mod_channels.get(interaction.guild_id)

        # follow up after the basic agree/disagree to first time or frequent offender
        if custom_id == "mod_agree":
            view = discord.ui.View(timeout=600)
            view.add_item(discord.ui.Button(label="First-Time Offender", style=discord.ButtonStyle.success, custom_id="agree_first"))
            view.add_item(discord.ui.Button(label="Repeat Offender", style=discord.ButtonStyle.primary, custom_id="agree_repeat"))
            await interaction.response.send_message("Please select the appropriate offender category:", view=view, ephemeral=True)

        elif custom_id == "mod_disagree":
            view = discord.ui.View(timeout=600)
            view.add_item(discord.ui.Button(label="First-Time Misreporter", style=discord.ButtonStyle.secondary, custom_id="disagree_first"))
            view.add_item(discord.ui.Button(label="Frequent Misreporter", style=discord.ButtonStyle.danger, custom_id="disagree_frequent"))
            await interaction.response.send_message("Please select the appropriate reporter category:", view=view, ephemeral=True)

        # handle the decisions according to flow chart
        elif custom_id == "agree_first" or custom_id == "agree_repeat":
            await interaction.response.send_message("Action recorded. Offending message deleted. Timeout applied.", ephemeral=True)
            
            # deleting the offending message
            try:
                data = self.last_reported_message
                guild = self.get_guild(data["guild_id"])
                channel = guild.get_channel(data["channel_id"])
                offending_message = await channel.fetch_message(data["message_id"])
                await offending_message.delete()
            except Exception as e:
                print(f"[ERROR] Could not delete offending message: {e}")
                await mod_channel.send("Error: Could not delete the offending message.")

            #messaging the author of the offending message
            user = self.last_reported_message["author"]

            try:
                if custom_id == "agree_first":
                    await user.send(
                        "Your account has been restricted on this server for one day due to offensive posts.\n"
                        "Please take a moment to consult with this resource: TO FILL IN"
                        "If you believe this was a mistake, you may respond to this message or contact a moderator."
                    )
                else:
                    await user.send(
                        "Your account has been banned from this server due to repeated offensive posts.\n"
                        "If you believe this was a mistake, you may respond to this message or contact a moderator."
                    )

            except discord.Forbidden:
                await mod_channel.send(f"Could not DM {user.name} due to their privacy settings.")

            if custom_id == "agree_first":
                await mod_channel.send("Moderator marked user as first-time offender. Post deleted and recieved one day timeout.")
            else:
                await mod_channel.send("Moderator marked user as repeat offender. Post deleted and user banned from server.")

        elif custom_id == "disagree_first":
            await interaction.response.send_message("Action recorded: Report dismissed. No further action required.", ephemeral=True)
            await mod_channel.send("Moderator disagreed with the report. No action taken.")

        elif custom_id == "disagree_frequent":
            await interaction.response.send_message("Action recorded: Reporter flagged for frequent misreporting.", ephemeral=True)
            await mod_channel.send("Reporter has been flagged as a frequent misreporter. Future reports may be suppressed.")

            #messaging the frequent misreporter
            try:
                reporter = self.last_reported_message["reporter"]
                await reporter.send(
                    "Your recent reports have been reviewed and deemed not offensive.\n"
                    "Please only use the reporting feature for genuine community violations. "
                    "Your account has internal flags restricting your behavior on this server."
                )
            except Exception as e:
                await mod_channel.send("Failed to notify the reporter via DM.")
                print(f"[DM ERROR] Could not message reporter: {e}")



    async def handle_dm(self, message):
        # Handle a help message
        if message.content == Report.HELP_KEYWORD:
            reply = "Use the `report` command to begin the reporting process.\n"
            reply += "Use the `cancel` command to cancel the report process.\n"
            await message.channel.send(reply)
            return

        author_id = message.author.id
        responses = []

        # Only respond to messages if they're part of a reporting flow
        if author_id not in self.reports and not message.content.startswith(Report.START_KEYWORD):
            return

        # If we don't currently have an active report for this user, add one
        if author_id not in self.reports:
            self.reports[author_id] = Report(self)

        # Let the report class handle this message; forward all the messages it returns to us
        responses = await self.reports[author_id].handle_message(message)
        for response in responses:
            if isinstance(response, tuple) and len(response) == 2:
                # If response is a tuple of (message, view)
                msg, view = response
                await message.channel.send(msg, view=view)
            else:
                # Regular string message
                await message.channel.send(response)

        # If the report is complete or cancelled, remove it from our map
        if self.reports[author_id].report_complete():
            self.reports.pop(author_id)

    async def handle_channel_message(self, message):
        # Only handle messages sent in the "group-#" channel
        if not message.channel.name == f'group-{self.group_num}':
            return

        # Forward the message to the mod channel
        mod_channel = self.mod_channels[message.guild.id]
        await mod_channel.send(f'Forwarded message:\n{message.author.name}: "{message.content}"')
        scores = self.eval_text(message.content)
        await mod_channel.send(self.code_format(scores))

    
    def eval_text(self, message):
        ''''
        TODO: Once you know how you want to evaluate messages in your channel, 
        insert your code here! This will primarily be used in Milestone 3. 
        '''
        return message

    
    def code_format(self, text):
        ''''
        TODO: Once you know how you want to show that a message has been 
        evaluated, insert your code here for formatting the string to be 
        shown in the mod channel. 
        '''
        return "Evaluated: '" + text+ "'"

    async def block_user(self, user_to_block, blocked_by):
        """Implement the logic to block a user"""
        try:
            await blocked_by.block(user_to_block)
            return True
        except discord.errors.HTTPException:
            return False


client = ModBot()
client.run(discord_token)