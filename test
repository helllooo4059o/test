import discord
import typing
import json
import re
import asyncio
import requests 
from aiohttp import ClientSession, ClientResponseError
from websockets import connect 
from datetime import datetime
import os
import subprocess
import glob
import shutil
import tempfile
import pynput
import pygetwindow as gw
import cv2
import numpy as np
from PIL import ImageGrab
import time
import os
import pytesseract
import mousekey as mk
import gc
import tkinter as tk
from tkinter import ttk
import sys
from collections import deque

LOCALVERSION = "1.0.1"

DISCORD_WS_BASE = "wss://gateway.discord.gg/?v=10&encoding-json"
PLACE_ID = 15532962292
BASE_ROBLOX_URL = f"https://www.roblox.com/games/{PLACE_ID}/"
SHARELINKS_API = "https://apis.roblox.com/sharelinks/v1/resolve-link"

ms_rblx_log_dir = os.path.expandvars(r"%LOCALAPPDATA%\Packages\ROBLOXCorporation.ROBLOX_55nm5eh3cm0pr\LocalState\logs")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

mkey = mk.MouseKey()
kb = pynput.keyboard.Controller()

REFERENCE_RES = (985, 768)
CROP_COORDS = (473, 667, 639, 705)

CONF_DIR = os.path.expandvars(r"%localappdata%\bazthedev\sniper")

DEFAULT_CONFIG = {
    "ROBLOSECURITY": "",
    "DISCORD_TOKEN": "",
    "WEBHOOK_URL": "",
    "PRESS_JOIN_BTN": False,
    "CHANNEL_FOR_SERVER": "",
    "IGNORE_CHANNELS": [],
    "IGNORE_USERS": [],
    "SCAN_CONTENT": True,
    "SCAN_EMBEDS": True,
    "BIOMES": {
        "SNOWY": False,
        "WINDY": False,
        "RAINY": False,
        "SAND STORM": False,
        "HELL": False,
        "STARFALL": False,
        "CORRUPTION": False,
        "NULL": False,
        "GLITCHED": True,
        "DREAMSPACE": False
    },
    "MERCHANTS": {
        "MARI": False,
        "JESTER": False
    },
    "VALIDATE_BIOME": True,
    "MENTION_ID": 0,
    "STAY_FOR_FULL_BIOME_LENGTH" : False,
    "SERVERS" : [],
    "SEND_VALID_BIOME_ALERT" : False,
    "LOG_BUT_NO_JOIN" : False,
    "DEBUG" : False,
    "PRESS_JOIN_TIMEOUT" : 180
}

if not os.path.exists(CONF_DIR):
    os.makedirs(CONF_DIR, exist_ok=True)

if not os.path.isfile(f"{CONF_DIR}/config.json"):
    x = open(f"{CONF_DIR}/config.json", "w")
    x.write("{}")
    x.close()
    with open(f"{CONF_DIR}/config.json", "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=4)
    print("Wrote default config")


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

KEYWORDS = {
    "SNOWY": ["snow", "snowy"],
    "WINDY": ["wind", "windy"],
    "RAINY": ["rain", "rainy"],
    "SAND STORM": ["sand", "storm", "sandstorm", "sand storm"],
    "HELL": ["hel", "hell", "blood", "bloodlust"],
    "STARFALL": ["starfall", "star fall"],
    "CORRUPTION": ["corrupt"],
    "NULL": ["nul", "null"],
    "GLITCHED": ["glitch", "glitched"],
    "DREAMSPACE": ["drea", "dream", "dreamspace", "dream space", "scape", "dream scape"],
    "MARI": ["mari", "void", "vc"],
    "JESTER": ["jester", "jes", "obli", "oblivion", "memory"]
}

BLACKLIST = [
    "end",
    "ended",
    "roll",
    "rolled",
    "aura",
    "macro",
    "snip",
    "snipe",
    "fak",
    "bait",
    "gone",
    "not",
    "owner",
    "pls",
    "plz",
    "need",
    "hunt",
    "want",
    "who",
    "have",
    "has",
    "was",
    "look",
    "real",
    "jk"
]

EMB_DATA = {
    "SNOWY": {"colour" : "#c4f5f6", "img_url" : "https://raw.githubusercontent.com/bazthedev/SolsScope/refs/heads/main/img/biome/SNOWY.png"},
    "WINDY": {"colour" : "#91f7ff", "img_url" : "https://raw.githubusercontent.com/bazthedev/SolsScope/refs/heads/main/img/biome/WINDY.png"},
    "RAINY": {"colour" : "#4385ff", "img_url" : "https://raw.githubusercontent.com/bazthedev/SolsScope/refs/heads/main/img/biome/RAINY.png"},
    "SAND STORM": {"colour" : "#f4c27c", "img_url" : "https://raw.githubusercontent.com/bazthedev/SolsScope/refs/heads/main/img/biome/SANDSTORM.png"},
    "HELL": {"colour" : "#4a1722", "img_url" : "https://raw.githubusercontent.com/bazthedev/SolsScope/refs/heads/main/img/biome/HELL.png"},
    "STARFALL": {"colour" : "#6784e0", "img_url" : "https://raw.githubusercontent.com/bazthedev/SolsScope/refs/heads/main/img/biome/STARFALL.png"},
    "CORRUPTION": {"colour" : "#9042ff", "img_url" : "https://raw.githubusercontent.com/bazthedev/SolsScope/refs/heads/main/img/biome/CORRUPTION.png"},
    "NULL": {"colour" : "#070910", "img_url" : "https://raw.githubusercontent.com/bazthedev/SolsScope/refs/heads/main/img/biome/NULL.png"},
    "GLITCHED": {"colour" : "#64fc64", "img_url" : "https://raw.githubusercontent.com/bazthedev/SolsScope/refs/heads/main/img/biome/GLITCHED.png"},
    "DREAMSPACE": {"colour" : "#ea6cbc", "img_url" : "https://raw.githubusercontent.com/bazthedev/SolsScope/refs/heads/main/img/biome/DREAMSPACE.png"},
    "MARI": {"colour" : "#ffffff", "img_url" : "https://static.wikia.nocookie.net/sol-rng/images/3/37/MARI_HIGH_QUALITYY.png/revision/latest"},
    "JESTER": {"colour" : "#b031ff", "img_url" : "https://static.wikia.nocookie.net/sol-rng/images/d/db/Headshot_of_Jester.png/revision/latest"}
}


def rgb2hex(r,g,b):
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

def hex2rgb(_hex):
    _hex = _hex.lstrip("#")
    if len(_hex) != 6:
        print(f"Invalid hex color format: '{_hex}'. Returning black.")
        return (0, 0, 0) 
    try:
        return tuple(int(_hex[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
         print(f"Invalid hex color value: '{_hex}'. Returning black.")
         return (0, 0, 0)
    
def is_position_stable(pos_history, tolerance_x=5, tolerance_y=15):
    if len(pos_history) < 2:
        return False
    x_vals = [p[0] for p in pos_history]
    y_vals = [p[1] for p in pos_history]
    return (max(x_vals) - min(x_vals) <= tolerance_x and
            max(y_vals) - min(y_vals) <= tolerance_y)

def get_roblox_window_bbox():
    windows = gw.getWindowsWithTitle("Roblox")
    for win in windows:
        if win.width > 100 and win.height > 100:
            return (win.left, win.top, win.width, win.height)
    return None

def capture_window(bbox, scale=0.5):
    x, y, w, h = bbox
    pil_image = ImageGrab.grab(bbox=(x, y, x + w, y + h))
    pil_image = pil_image.resize((int(w * scale), int(h * scale)))
    return pil_image, cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR), scale



def match_with_orb(window_img, template_kp, template_des, debug_img=None):
    orb = cv2.ORB_create(nfeatures=500)
    img_gray = cv2.cvtColor(window_img, cv2.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(img_gray, None)

    if des2 is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(template_des, des2)
    matches = [m for m in matches if m.distance < 64]

    if len(matches) < 8:
        return None

    points = np.float32([kp2[m.trainIdx].pt for m in matches])
    stddev = np.std(points, axis=0)
    if max(stddev) > 60:
        return None

    if debug_img is not None:
        for m in matches[:20]:
            pt1 = tuple(np.int32(template_kp[m.queryIdx].pt))
            pt2 = tuple(np.int32(kp2[m.trainIdx].pt))
            cv2.line(debug_img, pt1, pt2, (255, 0, 0), 1)

    avg_point = np.mean(points, axis=0)

    del img_gray, des2, kp2, matches, points, stddev
    gc.collect()

    return int(avg_point[0]), int(avg_point[1])


def match_with_ocr(pil_img, debug_img=None):
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    for i, text in enumerate(data["text"]):
        if text.strip().lower() == "start":
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            if debug_img is not None:
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            return x + w // 2, y + h // 2
    return None


def get_search_roi(window_img):
    h, w = window_img.shape[:2]
    roi_x1 = int(w * 0.35)
    roi_x2 = int(w * 0.65)
    roi_y1 = int(h * 0.75)
    roi_y2 = int(h * 0.95)
    roi = window_img[roi_y1:roi_y2, roi_x1:roi_x2]
    return roi, (roi_x1, roi_y1)


def load_orb_template(template_path):
    template = cv2.imread(template_path, 0)
    if template is None:
        raise ValueError("Template not found.")
    orb = cv2.ORB_create(nfeatures=500)
    kp, des = orb.detectAndCompute(template, None)
    return template, kp, des


async def wait_and_click_start_button(timeout=30, debug=True):
    print("Finding and clicking start button...")
    start_time = time.time()
    pos_history = deque(maxlen=5)

    template_path = resource_path("start_button.png")
    template, template_kp, template_des = load_orb_template(template_path)

    while time.time() - start_time < timeout:
        bbox = get_roblox_window_bbox()
        if not bbox:
            await asyncio.sleep(0.5)
            continue

        pil_img, window_img, scale = capture_window(bbox, scale=0.5)
        debug_img = window_img.copy() if debug else None

        roi_img, offset = get_search_roi(window_img)

        if debug and debug_img is not None:
            x1, y1 = offset
            x2 = x1 + roi_img.shape[1]
            y2 = y1 + roi_img.shape[0]
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug_img, "ROI", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("Debug ROI and Matches", debug_img)
            cv2.waitKey(50)


        pos = match_with_orb(roi_img, template_kp, template_des, debug_img=roi_img if debug else None)

        if not pos:
            if debug:
                print("[-] ORB failed, trying OCR...")
            pos = match_with_ocr(pil_img, debug_img=debug_img)
            if pos:
                pos = (pos[0], pos[1])
                pos = (int(pos[0] / scale), int(pos[1] / scale))
                pos_history.append(pos)
        else:
            pos = (pos[0] + offset[0], pos[1] + offset[1])
            pos = (int(pos[0] / scale), int(pos[1] / scale))
            pos_history.append(pos)

        if pos_history and is_position_stable(pos_history):
            x_screen = bbox[0] + pos_history[-1][0]
            y_screen = bbox[1] + pos_history[-1][1]
            mkey.left_click_xy_natural(x_screen, y_screen)
            print(f"[✔] Clicked Start button at ({x_screen}, {y_screen})")

            if debug and debug_img is not None:
                cv2.imshow("Debug Match", debug_img)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()

            return True
        else:
            if debug:
                print(f"[*] Position not stable yet: {list(pos_history)}")

        await asyncio.sleep(0.5)

    print("[x] Timeout reached.")
    return False

def get_latest_hovertext(logs_dir):
    log_files = glob.glob(os.path.join(logs_dir, "*.log"))
    if not log_files:
        return None

    latest_log_file = max(log_files, key=os.path.getctime)
    try:
        temp_file = os.path.join(tempfile.gettempdir(), "baz_sniper_biome_validation.log")
        shutil.copy2(latest_log_file, temp_file)
    except PermissionError:
        return None

    json_pattern = re.compile(r'\{.*\}')
    last_hover_text = None

    try:
        with open(temp_file, "r", encoding="utf-8", errors="ignore") as file:
            for line in reversed(file.readlines()):
                match = json_pattern.search(line)
                if match:
                    try:
                        json_data = json.loads(match.group())
                        hover_text = json_data.get("data", {}).get("largeImage", {}).get("hoverText")
                        if hover_text:
                            return hover_text
                    except json.JSONDecodeError:
                        continue
    except Exception:
        return None
    
    return last_hover_text

class Sniper:
    def __init__(self):
        self.config = self.load_config()
        self.validate_config()
        self.roblox_session: typing.Optional[ClientSession] = None
        self._refresh_task = None

        self.is_running = False
        self.in_biome = False

        self.link_pattern = re.compile(
            f"https://www.roblox.com/games/{PLACE_ID}/*\\?privateServerLinkCode="
        )
        self.link_pattern_2 = re.compile(r"https://.*&type=Server")

        self.webhook = ""

    def load_config(self):
        with open(f"{CONF_DIR}/config.json", "r") as f:
            return json.load(f)
    
    def validate_config(self):
        for k in DEFAULT_CONFIG.keys():
            if k not in self.config.keys():
                self.config[k] = DEFAULT_CONFIG[k]

        for k in self.config.keys():
            if k not in DEFAULT_CONFIG.keys():
                del self.config[k]

        self.save_config()
    
    def save_config(self):
        with open(f"{CONF_DIR}/config.json", "w") as f:
            json.dump(self.config, f, indent=4)

    def load_toggles(self):
        
        biomes = self.config.get("BIOMES", {})
        merchants = self.config.get("MERCHANTS", {})

        enabled_biomes = []
        enabled_merchants = []

        for biome in biomes.keys():
            if biomes[biome]:
                enabled_biomes.append(biome)

        for merchant in merchants.keys():
            if merchants[merchant]:
                enabled_merchants.append(merchant)

        return enabled_biomes, enabled_merchants

    async def setup(self):
        """Initializes the aiohttp session with the Roblox security cookie."""
        roblo_key = self.config.get("ROBLOSECURITY", "")
        if not roblo_key:
            print("Error: ROBLOSECURITY is missing in settings. Cannot setup sniper.")
            return False

        self.roblox_session = ClientSession()
        self.roblox_session.cookie_jar.update_cookies({".ROBLOSECURITY": roblo_key})
        self.roblox_session.headers["User-Agent"] = "GLITCHED Sniper"
        print("Roblox session initialized.")
        return True

    async def _identify(self, ws):
        """Sends the IDENTIFY payload to the Discord gateway."""
        discord_token = self.config.get("DISCORD_TOKEN", "")
        if not discord_token:
            print("Error: DISCORD_TOKEN missing. Cannot identify.")
            raise ConnectionAbortedError("Discord token missing") 

        try:
            identify_payload = {
                "op": 2, 
                "d": {
                    "token": discord_token,
                    "properties": {"$os": "windows", "$browser": "chrome", "$device": "pc"},

                }
            }
            await ws.send(json.dumps(identify_payload))
            print("Identify payload sent.")
        except Exception as e:
            print(f"Error sending IDENTIFY: {e}")
            raise 


    async def _subscribe(self, ws):
        """Sends the GUILD_SUBSCRIBE payload (Opcode 14 - undocumented?)."""

        try:
            subscription_payload = {
                "op": 14, 
                "d": {
                    "guild_id": self.config.get("SERVERS", [])[0],
                }
            }
            await ws.send(json.dumps(subscription_payload))
            print(f"Subscribe payload sent for Guild {self.config.get('SERVERS', [])[0]}.")
        except Exception as e:
            print(f"Error sending SUBSCRIBE: {e}")
            raise

    async def heartbeat(self, ws, interval):
        """Sends heartbeat payloads at the specified interval."""
        while self.is_running:
                try:
                    heartbeat_payload = {
                        'op': 1, 
                        'd': None 
                    }
                    await ws.send(json.dumps(heartbeat_payload))

                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    print("Heartbeat task cancelled.")
                    break
                except Exception as e:

                    if self.is_running: 
                        print(f"Error in heartbeat loop: {e}")

                    self.is_running = False
                    break
        print("Heartbeat loop stopped.")

    async def _on_message(self, ws):
        """Listens for and processes messages from the Discord gateway."""
        while self.is_running:
                try:
                    raw_event = await ws.recv()
                    event = json.loads(raw_event)

                    op = event.get('op')
                    event_type = event.get('t')
                    data = event.get('d')

                    if op == 10: 
                        interval = data["heartbeat_interval"] / 1000

                        asyncio.create_task(self.heartbeat(ws, interval))
                        print(f"Received Hello. Heartbeat interval: {interval}s")

                        await self._identify(ws)

                    elif op == 11: 

                        pass

                    elif op == 0: 
                        if event_type == "READY":

                            print(f"Received READY. Session ID: {data.get('session_id')}")

                            await self._subscribe(ws)

                        elif event_type == "MESSAGE_CREATE":
                            if not self.in_biome:
                                asyncio.create_task(self.process_message(data))

                    elif op == 7: 
                        print("Received Reconnect request from Discord.")
                        self.is_running = False 
                        break 

                    elif op == 9: 
                        print(f"Received Invalid Session (Op 9). Can resume: {data}")

                        self.is_running = False
                        break 

                except asyncio.CancelledError:
                    print("Message listener task cancelled.")
                    break
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from gateway: {raw_event[:100]}...") 
                    continue 
                except Exception as e:
                    if self.is_running:
                        print(f"Error in message listener: {e}")
                    self.is_running = False 
                    break 

        print("Message listener loop stopped.")


    async def _extract_server_code(self, message: str) -> typing.Optional[str]:
        """Extracts the private server link code from a message, resolving share links if necessary."""

        if link_match := self.link_pattern.search(message):
            try:
                code = link_match.group(0).split("LinkCode=")[-1]
                print(f"Extracted direct link code: {code[:5]}...")
                return code
            except IndexError:
                 print(f"Error parsing direct link code from: {link_match.group(0)}")
                 return None

        if link_match_2 := self.link_pattern_2.search(message):
            try:
                share_code = link_match_2.group(0).split("code=")[-1].split("&")[0]
                print(f"Found share link code: {share_code}. Attempting conversion...")
                return await self._convert_link(share_code)
            except IndexError:
                 print(f"Error parsing share link code from: {link_match_2.group(0)}")
                 return None

        return None
    
    async def _extract_server_code_embed(self, emb : dict) -> typing.Optional[str]:
        """Extracts the private server link code from a message, resolving share links if necessary."""

        emb_title = emb.get("title", "")
        description = emb.get("description", "")
        fields = emb.get("fields", [])

        if link_match := self.link_pattern.search(description):
            try:
                code = link_match.group(0).split("LinkCode=")[-1]
                print(f"Extracted direct link code: {code[:5]}...")
                return code
            except IndexError:
                 print(f"Error parsing direct link code from: {link_match.group(0)}")
                 return None

        if link_match_2 := self.link_pattern_2.search(description):
            try:
                share_code = link_match_2.group(0).split("code=")[-1].split("&")[0]
                print(f"Found share link code: {share_code}. Attempting conversion...")
                return await self._convert_link(share_code)
            except IndexError:
                 print(f"Error parsing share link code from: {link_match_2.group(0)}")
                 return None
            
        for field in fields:
            value = field["value"]
            if link_match := self.link_pattern.search(value):
                try:
                    code = link_match.group(0).split("LinkCode=")[-1]
                    print(f"Extracted direct link code: {code[:5]}...")
                    return code
                except IndexError:
                    print(f"Error parsing direct link code from: {link_match.group(0)}")
                    return None

            if link_match_2 := self.link_pattern_2.search(value):
                try:
                    share_code = link_match_2.group(0).split("code=")[-1].split("&")[0]
                    print(f"Found share link code: {share_code}. Attempting conversion...")
                    return await self._convert_link(share_code)
                except IndexError:
                    print(f"Error parsing share link code from: {link_match_2.group(0)}")
                    return None

        return None
    
    async def wait_for_biome_end(self, biome_name):
        
        self.in_biome = True

        print(f"Waiting for duration of {biome_name}")

        while self.in_biome:

            valid_biome = await self.validate_biome(biome_name)

            if not valid_biome:
                print("Biome ended, leaving")
                await self.leave_server()
                await asyncio.sleep(5)
                self.in_biome = False

            await asyncio.sleep(0.2)

    async def validate_biome(self, biome_name):

        while True:
        
            current_biome = get_latest_hovertext(ms_rblx_log_dir)

            if current_biome is None:
                continue

            if current_biome.lower() == biome_name.lower():
                return True
            else:
                return False


    async def _convert_link(self, link_id: str) -> typing.Optional[str]:
        """Converts a roblox.com/share link code to a privateServerLinkCode using Roblox API."""
        if not self.roblox_session:
            print("Cannot convert link: Roblox session not initialized.")
            return None

        payload = {"linkId": link_id, "linkType": "Server"}
        headers = {"Referer": BASE_ROBLOX_URL} 

        try:
            async with self.roblox_session.post(SHARELINKS_API, json=payload, headers=headers) as response:

                if response.status == 403 and "X-CSRF-TOKEN"in response.headers:
                    csrf_token = response.headers["X-CSRF-TOKEN"]
                    print("Received CSRF token. Retrying link conversion...")
                    self.roblox_session.headers["X-CSRF-TOKEN"] = csrf_token

                    async with self.roblox_session.post(SHARELINKS_API, json=payload, headers=headers) as retry_response:
                        retry_response.raise_for_status() 
                        data = await retry_response.json()
                else:
                    response.raise_for_status() 
                    data = await response.json()

            invite_data = data.get("privateServerInviteData")
            if not invite_data:
                 print(f"Invalid response format from sharelinks API: {data}")
                 return None

            if invite_data.get("placeId") != PLACE_ID:
                print(f"Filtered non-Sols link! Place ID: {invite_data.get('placeId')}")
                return None

            link_code = invite_data.get("linkCode")
            if link_code:
                 print(f"Converted share link. Private Server Link Code: {link_code[:5]}...")
                 return link_code
            else:
                 print(f"'linkCode' not found in sharelinks API response: {data}")
                 return None

        except ClientResponseError as e:
             print(f"HTTP Error converting share link ({e.status}): {e.message}")
        except asyncio.TimeoutError:
             print(f"Timeout converting share link.")
        except Exception as e:
            print(f"Unexpected error converting share link: {str(e)}")
        return None
    
    async def send_valid_biome(self, biome, valid=True):

        if not self.webhook:
            print("Cannot send notification: Main webhook not available.")
            return

        colour = hex2rgb(EMB_DATA[biome]["colour"])

        if valid:
            description = f"A **valid** {biome} server was found!"

            embed = discord.Embed(
                title = f'[{datetime.now().strftime("%H:%M:%S")}] {biome} was validated!',
                description = description,
                colour = discord.Colour.from_rgb(*colour),
                timestamp=datetime.now()
            )
            embed.set_thumbnail(url="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/Eo_circle_green_checkmark.svg/1200px-Eo_circle_green_checkmark.svg.png")
        else:
            description = f"The biome {biome} was checked and does not match the active biome within the server."

            embed = discord.Embed(
                title = f'[{datetime.now().strftime("%H:%M:%S")}] {biome} does not match.',
                description = description,
                colour = discord.Colour.from_rgb(*colour),
                timestamp=datetime.now()
            )
            embed.set_thumbnail(url="https://mfcfoundation.co.uk/wp-content/uploads/2017/06/RED-CROSS-300x300.png")

        embed.set_footer(text=f"baz's sigma sniper v{LOCALVERSION}")

        try:
            self.webhook.send(
                embed=embed
            )
            print(f"Sent validation notification for {biome} link.")

        except Exception as e:
            print(f"Error sending validation notification webhook: {e}")

    async def _handle_server_join(self, match_type : str, server_code: str):
        """Initiates the server join and sets the sniped event."""

        join_successful = self.join_private_server_link(server_code)
        self.in_biome = True

        if join_successful:
            print(f"Join command issued for {match_type} server.")
            print("Waiting for Roblox window and Sol's to load data...")
            await asyncio.sleep(20)

            if self.config.get("PRESS_JOIN_BTN", False):
                join_serv = await wait_and_click_start_button(timeout=self.config.get("PRESS_JOIN_TIMEOUT", 180), debug=self.config.get("DEBUG", True))
                if not join_serv:
                    if await self.validate_biome(match_type):
                        print("Biomes match!")
                        await self.send_valid_biome(match_type, True)
                    else:
                        print("Biomes do not match, leaving and restarting sniper...")
                        await self.leave_server()
                        await asyncio.sleep(5)
                        self.in_biome = False
                        await self.send_valid_biome(match_type, False)
                        return
            
            await asyncio.sleep(2)

            if match_type.lower() != "mari" and match_type.lower() != "jester":

                if self.config.get("VALIDATE_BIOME", False):
                    if await self.validate_biome(match_type):
                        print("Biomes match!")
                        await self.send_valid_biome(match_type, True)
                    else:
                        print("Biomes do not match, leaving")
                        await self.leave_server()
                        await asyncio.sleep(5)
                        self.in_biome = False
                        await self.send_valid_biome(match_type, False)
                        return

                if self.config.get("STAY_FOR_FULL_BIOME_LENGTH", False):
                    await self.wait_for_biome_end(match_type)
            else:
                print("Waiting 4 minutes before resuming sniping...")
                await asyncio.sleep(240)

        else:
             print(f"Failed to issue join command for {match_type} server.")

        self.in_biome = False

    async def leave_server(self):
        kb.press(pynput.keyboard.Key.esc)
        await asyncio.sleep(0.4)
        kb.release(pynput.keyboard.Key.esc)
        await asyncio.sleep(0.4)
        kb.press("l")
        await asyncio.sleep(0.4)
        kb.release("l")
        await asyncio.sleep(0.4)
        kb.press(pynput.keyboard.Key.enter)
        await asyncio.sleep(0.4)
        kb.release(pynput.keyboard.Key.enter)
        await asyncio.sleep(2)

    async def _send_notification(self, match_type: str, server_code: str, msg_content: str, uses_embed : bool, channel_id : int):
        """Sends a notification embed to the primary webhook and forwards it."""
        if not self.webhook:
            print("Cannot send notification: Main webhook not available.")
            return

        colour = hex2rgb(EMB_DATA[match_type]["colour"])

        embed_link = f"{BASE_ROBLOX_URL}?privateServerLinkCode={server_code}"

        description = f"**Link:** [Click to Join]({embed_link})\n**Code:** `{server_code}`"

        if uses_embed:
            description += "\n**Macro:** :white_check_mark:"
        else:
            description += "\n**Macro:** :x:"

        description += f"\n**Channel:** <#{str(channel_id)}>"

        embed = discord.Embed(
            title = f'[{datetime.now().strftime("%H:%M:%S")}] {match_type} Link Sniped!',
            description = description,
            colour = discord.Colour.from_rgb(*colour),
            timestamp=datetime.now()
        )
        embed.set_thumbnail(url=EMB_DATA[match_type]["img_url"])

        original_msg_display = msg_content[:1000] + ("..."if len(msg_content) > 1000 else "")
        embed.add_field(name = "Original Message", value = f"```{original_msg_display}```", inline=False)
        embed.set_footer(text=f"baz's sigma sniper v{LOCALVERSION}")

        content_ping = f"<@{self.config['MENTION_ID']}>"if self.config.get("MENTION_ID", 0) else ""

        try:
            self.webhook.send(
                content=content_ping,
                embed=embed
            )
            print(f"Sent notification for {match_type} link.")

        except Exception as e:
            print(f"Error sending notification webhook: {e}")

    async def match_pattern(self, content):

        m = content.lower()

        for entity in KEYWORDS:
            for kw in KEYWORDS[entity]:
                if kw in m:
                    hit = False
                    for word in BLACKLIST:
                        if word in m:
                            hit = True
                            break
                        
                    if not hit:
                        return entity
        
        return None
    

    async def process_message(self, data) -> None:
        """Processes a single message content for sniping."""

        content = data.get("content", "")
        embeds = data.get("embeds", [])
        author_data = data.get("author", {})
        is_bot = author_data.get("bot", False)
        guild_id = data.get("guild_id", 0)
        channel_id = data.get("channel_id", 0)

        kw_match = None
        emb_match = None

        try:

            if int(guild_id) not in self.config.get("SERVERS", []):
                return

            if int(channel_id) in self.config.get("IGNORE_CHANNELS", []):
                return
            
            if int(author_data.get("id", 0)) in self.config.get("IGNORE_USERS", []):
                return
            
            
            if self.config.get("SCAN_CONTENT", False):
                kw_match = await self.match_pattern(content)

            if self.config.get("SCAN_EMBEDS", False) and is_bot:
                try:
                    to_process = embeds[0]
                except IndexError:
                    return
                
                emb_match = await self.match_pattern(to_process.get("title", ""))

            if not kw_match and not emb_match:
                return
            
            biomes, merchants = self.load_toggles()

            if kw_match:
                if (kw_match not in biomes and kw_match not in merchants):
                    return
            
            if emb_match:
                if (emb_match not in biomes and emb_match not in merchants):
                    return

            if kw_match:
                server_code = await self._extract_server_code(content)
                if not server_code:
                    return
                print(f"!!! Valid {kw_match} message found! Code: {server_code[:5]}... !!!")
                print(f"Original content: {content[:100]}...")
                await self._send_notification(kw_match, server_code, content, False, channel_id)
                if not self.config.get("LOG_BUT_NO_JOIN", False):
                    await self._handle_server_join(kw_match, server_code)
            elif emb_match:
                server_code = await self._extract_server_code_embed(to_process)
                if not server_code:
                    server_code = await self._extract_server_code(content)
                    if not server_code:
                        return

                print(f"!!! Valid {emb_match} message found! Code: {server_code[:5]}... !!!")
                print(f"Original content: {to_process['title']}...")

                await self._send_notification(emb_match, server_code, to_process["title"], True, channel_id)
                if not self.config.get("LOG_BUT_NO_JOIN", False):
                    await self._handle_server_join(emb_match, server_code)

        except Exception as e:
            print(f"Error processing message: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def join_private_server_link(self, server_code):
        """Attempts to join a Roblox private server using the server code via protocol handler."""
        if not server_code:
            print("Attempted to join private server with empty code.")
            return False
        try:

            final_link = f"roblox://placeID={PLACE_ID}^&linkCode={server_code}"
            print(f"Attempting to join server with link code: {server_code[:5]}... (using {final_link[:30]}...)")

            subprocess.Popen(["start", "", final_link], shell=True) 
            print("Issued command to join server.")
            return True
        except Exception as e:
            print(f"Error executing command to join private server: {e}")
            return False

    async def run(self):
        """Main execution loop for the sniper instance."""
        self.is_running = True
        print("Starting sniper run...")

        print("Reloading config...")

        self.config = self.load_config()
        
        self.webhook = discord.SyncWebhook.from_url(self.config.get("WEBHOOK_URL", ""))

        if not self.config.get("ROBLOSECURITY") or not self.config.get("DISCORD_TOKEN"):
            print("ROBLOSECURITY or DISCORD_TOKEN missing. Exiting run.")
            self.is_running = False
            return

        biome_toggles, merchant_toggles = self.load_toggles()

        if len(biome_toggles) == 0 and len(merchant_toggles) == 0:
            print("No snipe types (Glitched/Dreamspace) enabled. Exiting run.")
            self.is_running = False
            return

        if not await self.setup():
            print("Failed initial setup (Roblox Session). Exiting run.")
            self.is_running = False
            return
        
        print(f"Starting connection loop. Monitoring for the following toggles:")
        print(str(biome_toggles))
        print(str(merchant_toggles))
        self.is_running = True
        while self.is_running:
            try:
                async with connect(DISCORD_WS_BASE, max_size=None, ping_interval=None) as ws:
                    print("WebSocket connected.")

                    await self._on_message(ws)

            except ConnectionRefusedError:
                print("Connection refused. Retrying in 15s...")
            except ConnectionAbortedError as e:
                print(f"Connection aborted ({e}). Retrying in 15s...")
            except Exception as e:
                print(f"Unexpected WebSocket error: {e}. Retrying in 15s...")
                import traceback
                print(traceback.format_exc())
            
            await asyncio.sleep(15)

        self.is_running = False
        print("Stop event received. Cleaning up...")
        if self.roblox_session and not self.roblox_session.closed:
            await self.roblox_session.close()
            print("Roblox session closed.")
        print("Sniper stopped.")

class ConfigGUI:
    def __init__(self, root, config):
        self.root = root
        self.root.title(f"baz's sigma sniper v{LOCALVERSION}")
        self.root.protocol("WM_DELETE_WINDOW",  sys.exit)
        self.config = config
        self.widgets = {}

        icon_path = os.path.join(CONF_DIR, "icon.ico")
        icon_url = "https://raw.githubusercontent.com/bazthedev/SolsScope/main/icon.ico"
        if not os.path.exists(icon_path):
            print("Icon file missing, attempting download...")
            try:
                dl = requests.get(icon_url, timeout=10)
                dl.raise_for_status()
                with open(icon_path, "wb") as f:
                    f.write(dl.content)
                print("Icon downloaded successfully.")
            except requests.RequestException as e:
                print(f"Failed to download icon: {e}") 
            except OSError as e:
                print(f"Error saving icon file: {e}") 
            except Exception as e:
                print(f"Unexpected error downloading/saving icon: {e}")

        if os.path.exists(icon_path):
            self.root.iconbitmap(icon_path)

        self.build_gui()
        ttk.Button(root, text="Save and Start Sniper", command=self.save_config).pack(pady=10)

    def build_gui(self):
        frame = ttk.Frame(self.root)
        frame.pack(padx=10, pady=10)

        row = 0
        for key, value in self.config.items():
            ttk.Label(frame, text=key).grid(row=row, column=0, sticky="nw")

            if isinstance(value, bool):
                var = tk.BooleanVar(value=value)
                self.widgets[key] = var
                ttk.Checkbutton(frame, variable=var).grid(row=row, column=1, sticky="w")

            elif isinstance(value, (int, str)):
                var = tk.StringVar(value=str(value))
                self.widgets[key] = var
                ttk.Entry(frame, textvariable=var).grid(row=row, column=1, sticky="w")

            elif isinstance(value, list):
                list_frame = ttk.Frame(frame)
                list_frame.grid(row=row, column=1, sticky="w")

                scrollbar = ttk.Scrollbar(list_frame)
                listbox = tk.Listbox(list_frame, height=5, width=40, selectmode=tk.SINGLE, yscrollcommand=scrollbar.set)
                scrollbar.config(command=listbox.yview)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

                for item in value:
                    listbox.insert(tk.END, str(item))

                btn_frame = ttk.Frame(frame)
                btn_frame.grid(row=row, column=2, sticky="n")

                entry_var = tk.StringVar()
                entry = ttk.Entry(btn_frame, textvariable=entry_var, width=10)
                entry.pack(pady=2)

                def add_item(lb=listbox, ev=entry_var):
                    item = ev.get()
                    if item:
                        lb.insert(tk.END, item)
                        ev.set("")

                def remove_selected(lb=listbox):
                    selection = lb.curselection()
                    if selection:
                        lb.delete(selection[0])

                ttk.Button(btn_frame, text="+", command=add_item).pack(pady=1)
                ttk.Button(btn_frame, text="-", command=remove_selected).pack(pady=1)

                self.widgets[key] = listbox

            elif isinstance(value, dict):
                sub_frame = ttk.LabelFrame(frame, text=key)
                sub_frame.grid(row=row, column=1, sticky="w", pady=5)
                self.widgets[key] = {}
                sub_row = 0
                for subkey, subval in value.items():
                    ttk.Label(sub_frame, text=subkey).grid(row=sub_row, column=0, sticky="w")
                    var = tk.BooleanVar(value=subval)
                    self.widgets[key][subkey] = var
                    ttk.Checkbutton(sub_frame, variable=var).grid(row=sub_row, column=1)
                    sub_row += 1

            row += 1

    def save_config(self):
        new_config = {}
        for key, widget in self.widgets.items():
            if isinstance(widget, tk.StringVar):
                val = widget.get()
                new_config[key] = int(val) if val.isdigit() else val

            elif isinstance(widget, tk.BooleanVar):
                new_config[key] = widget.get()

            elif isinstance(widget, tk.Listbox):
                items = widget.get(0, tk.END)
                try:
                    new_config[key] = [int(item) for item in items]
                except ValueError:
                    new_config[key] = list(items)

            elif isinstance(widget, dict):
                new_config[key] = {k: v.get() for k, v in widget.items()}

        with open(f"{CONF_DIR}/config.json", "w") as f:
            json.dump(new_config, f, indent=4)

        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    sniper = Sniper()
    app = ConfigGUI(root, sniper.load_config())
    root.mainloop()
    asyncio.run(sniper.run())
