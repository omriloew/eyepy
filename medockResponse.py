from dataclasses import dataclass
import socket
import struct
import time
import binascii
from psychopy import event

COMMAND_TO_ID = {
    'GET_STATUS': 0,
    'SELECT_TP': 1,
    'START': 2,
    'PAUSE': 3,
    'TRIGGER': 4,
    'STOP': 5,
    'ABORT': 6,
    'YES': 7,        # להתחיל להעלות טמפ'
    'NO': 8,         # להתחיל להוריד טמפ'
    'COVAS': 9,
    'VAS': 10,
    'SPECIFY_NEXT': 11,
    'T_UP': 12,      # להעלות ב-Δ°C (נשלח כ-Δ*100)
    'T_DOWN': 13,    # להוריד ב-Δ°C (נשלח כ-Δ*100)
    'KEYUP': 14,     # לעצור גרדיאנט
    'UNNAMED': 15
}
ID_TO_COMMAND = {v: k for k, v in COMMAND_TO_ID.items()}

TEST_STATES = {0: 'IDLE', 1: 'RUNNING', 2: 'PAUSED', 3: 'READY'}
SYSTEM_STATES = {0: 'IDLE', 1: 'READY', 2: 'TEST IN PROGRESS'}
RESPONSE_CODES = {
    0: "OK",
    1: "FAIL: ILLEGAL PARAMETER",
    2: "FAIL: ILLEGAL STATE",
    3: "FAIL: NOT THE PROPER TEST STATE",
    4096: "DEVICE COMMUNICATION ERROR",
    8192: "SAFETY WARNING (continues)",
    16384: "SAFETY ERROR (goes to IDLE)"
}
def check_for_escape():
    keys = event.getKeys(keyList=['escape'])
    return 'escape' in keys

def _u32(n: int) -> bytes:
    # big-endian (רשת)
    return struct.pack('!I', n)

def _i16(n: int) -> bytes:
    return struct.pack('!h', n)

def _now_u32() -> int:
    return int(time.time())

def _build_command(cmd, param=None) -> bytes:
    """
    פורמט מסר (כפי שמופיע בדוגמה):
    [LEN:4][TIMESTAMP:4][CMD_ID:1][PARAM?:4]
    - LEN = אורך החלק שאחרי ה-LEN (כלומר LEN = 5 או 9)
    - TIMESTAMP: שניות UNIX (u32)
    - CMD_ID: בית אחד
    - PARAM: אם יש פרמטר → u32 (רשת)
      * אם הפרמטר float של °C → מכפילים ב-100 ושולחים כמספר שלם
      * אם זה מחרוזת בינארית של תוכנית (e.g. '00011100') → int(base=2)
    """
    if isinstance(cmd, str):
        cmd_id = COMMAND_TO_ID[cmd.upper()]
    else:
        cmd_id = int(cmd)

    param_bytes = b''
    if param is not None:
        if isinstance(param, str):
            # למשל '00011100'
            p = int(param, 2)
        elif isinstance(param, float):
            # 37.5°C → 3750
            p = int(round(param * 100))
        else:
            p = int(param)
        param_bytes = _u32(socket.htonl(p))

    payload = b''.join([
        _u32(socket.htonl(_now_u32())),
        bytes([cmd_id]),
        param_bytes
    ])
    # LEN הוא האורך של payload בלבד (ללא 4 הבייטים של ה-LEN עצמו)
    msg = _u32(len(payload)) + payload
    return msg

@dataclass
class MedocResponse:
    length: int
    timestamp: int
    command_id: str
    system_state: str
    test_state: str
    resp_code: str
    test_time_s: float
    temperature_c: float
    covas: int
    yes: int
    no: int
    message: bytes

    @classmethod
    def parse(cls, buf: bytes) -> "MedocResponse":
        """
        פירוק תשובה לפי הדוגמה (שילוב המידע מהקטעים שהבאת):
        Offsetים (לאחר תחילת ההודעה, כולל כותרת LEN):
        0..4   length (u32)      ← בדוגמה המקורית השתמשו בטעות ב-H; כאן מתקנים ל-!I
        4..8   timestamp (u32)
        8      command (u8)
        9      system_state (u8)
        10     test_state (u8)
        11..13 resp_code (u16)
        13..17 test_time_ms (u32) → שניות = /1000
        17..19 temperature (i16)  → °C = /100
        19     COVAS (u8)
        20     yes (u8)
        21     no (u8)
        22..   message (bytes עד length)
        """
        if len(buf) < 22:
            raise ValueError(f"buffer too short ({len(buf)})")
        # הכל בליטל־אנדיאן
        length = struct.unpack_from('<I', buf, 0)[0]  # 0..4
        timestamp = struct.unpack_from('<I', buf, 4)[0]  # 4..8 (שניות UNIX)
        command_id = buf[8]  # 8
        system_state = buf[9]  # 9
        test_state = buf[10]  # 10
        resp_code = struct.unpack_from('<H', buf, 11)[0]  # 11..13
        test_time_ms = struct.unpack_from('<I', buf, 13)[0]  # 13..17
        temperature = struct.unpack_from('<h', buf, 17)[0]  # 17..19 (×0.01°C)
        covas = buf[19]  # 19
        yes = buf[20]  # 20
        no = buf[21]  # 21
        message = buf[22:22 + max(0, length - 22)]  # 22..

        return cls(
                length=length,
                timestamp=timestamp,
                command_id=ID_TO_COMMAND.get(command_id, f"CMD({command_id})"),
                system_state=SYSTEM_STATES.get(system_state, f"UNKNOWN({system_state})"),
                test_state=TEST_STATES.get(test_state, f"UNKNOWN({test_state})"),
                resp_code=RESPONSE_CODES.get(resp_code, f"CODE {resp_code}"),
                test_time_s=test_time_ms / 1000.0,
                temperature_c=temperature / 100.0,
                covas=covas, yes=yes, no=no, message=message
            )

    def __str__(self):

        lines = [
            f"timestamp : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}",
            f"command   : {self.command_id}",
            f"state     : {self.system_state}",
            f"testState : {self.test_state}",
            f"respCode  : {self.resp_code}",
            f"temp (°C) : {self.temperature_c:.2f}",
        ]
        if self.test_state == 2:  # TEST IN PROGRESS
            lines.append(f"test time : {self.test_time_s:.3f}s")
        if self.message:
            try:
                lines.append(f"message   : {self.message.decode('utf-8', 'ignore')}")
            except:
                lines.append(f"message(hex): {binascii.hexlify(self.message).decode()}")
        if self.yes: lines.append("YES pressed")
        if self.no:  lines.append("NO pressed")
        return "\n".join(lines)