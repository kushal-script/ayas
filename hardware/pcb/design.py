"""Ayas Phase 2 electronics: one Python description per board (docs/hardware/pcb_phase2.md).

Each board is a list of parts and a net map; gen_sch.py and gen_pcb.py build the KiCad schematic and
PCB from it, so the two always agree. Pins are referenced by NAME (every pin of that name on the part
joins the net, e.g. all GND pins) or by '#<number>' for a specific pin.

Datasheet sources for the power parts (copies read 24 Nov 2026):
  STUSB4500  ST DS12499 rev 8, Table 1 (pins), §2.2, §6.2 Fig. 9 (minimum implementation)
  TPS565201  TI SLVSE71, Pin Functions, Eq. 2 (VOUT = 0.760 (1 + R1/R2)), Table 2 (5 V: 54.9k/10k, 3.3 uH, 20-68 uF)
  INA226     TI SBOS547C, Table 4-1 (pins), VS 2.7-5.5 V, A0 = A1 = GND -> 0x40
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Part:
    ref: str
    symbol: str                 # 'Library:Name' in the KiCad symbol libraries
    footprint: str              # 'Library:Name' in the KiCad footprint libraries
    value: str
    pins: dict = field(default_factory=dict)   # pin name or '#num' -> net
    mpn: str = ''
    note: str = ''
    xy: tuple | None = None     # PCB placement hint (mm, board frame), set in layout.py


def R(ref, value, a, b, fp='Resistor_SMD:R_0603_1608Metric', mpn=''):
    return Part(ref, 'Device:R', fp, value, {'#1': a, '#2': b}, mpn)


def C(ref, value, a, b, fp='Capacitor_SMD:C_0603_1608Metric', mpn=''):
    return Part(ref, 'Device:C', fp, value, {'#1': a, '#2': b}, mpn)


LINK = {'#1': 'V12_LED', '#2': 'V12_LED', '#3': 'GND', '#4': 'GND', '#5': 'V5', '#6': '3V3', '#7': 'SDA', '#8': 'SCL',
        '#9': 'ALERT_N', '#10': 'GND', '#11': 'BTN_N', '#12': 'HOLD'}     # POWER J2 <-> LIGHT/IO J1


def jst_ph(ref, nets, note=''):
    n = len(nets)
    return Part(ref, 'Connector_Generic:Conn_01x%02d' % n,
                'Connector_JST:JST_PH_B%dB-PH-K_1x%02d_P2.00mm_Vertical' % (n, n), 'JST-PH %d' % n,
                {'#%d' % (i + 1): net for i, net in enumerate(nets)}, 'B%dB-PH-K-S (JST)' % n, note)


def jst_sh(ref, nets, note=''):
    """JST-SH 1 mm side-entry: 4.25 mm tall (fits under the display posts), 1 A per contact."""
    n = len(nets)
    return Part(ref, 'Connector_Generic:Conn_01x%02d' % n,
                'Connector_JST:JST_SH_SM%02dB-SRSS-TB_1x%02d-1MP_P1.00mm_Horizontal' % (n, n), 'JST-SH %d' % n,
                {'#%d' % (i + 1): net for i, net in enumerate(nets)}, 'SM%02dB-SRSS-TB (JST)' % n, note)


def uv_pins(ch='UV', ntc='NTC'):
    """Flex pinout, a palindrome so the link works whichever way round the flex is made or inserted."""
    k = 'K_' + ch
    return ['GND', 'V12_LED', k, ntc, ntc, k, 'V12_LED', 'GND']


def ffc8(ref, nets, note=''):
    return Part(ref, 'Connector_Generic:Conn_01x08', 'Connector_FFC-FPC:Molex_200528-0080_1x08-1MP_P1.00mm_Horizontal',
                'FFC 8 ZIF', {'#%d' % (i + 1): n for i, n in enumerate(nets)}, '200528-0080 (Molex)', note)


def POD_PINS(side):
    """Pod dock contacts (body side), one pod: 12 V, white, UV, NTC, GND."""
    return ['GND', 'V12_LED', 'K_WHITE_' + side, 'K_UV_' + side, 'NTC_' + side]


POD_FLEX = ['GND', 'V12_LED', 'K_UV', 'K_WHITE', 'NTC', 'NTC', 'K_WHITE', 'K_UV', 'V12_LED', 'GND']   # palindrome
HOOD_PADS = ['GND', 'V12_LED', 'K_RING1', 'K_RING2', 'K_RING3', 'K_RING4', 'K_RING5', 'K_RING6', 'HOOD_N']


def ffc10(ref, nets, note=''):
    return Part(ref, 'Connector_Generic:Conn_01x10', 'Connector_FFC-FPC:Molex_200528-0100_1x10-1MP_P1.00mm_Horizontal',
                'FFC 10 ZIF', {'#%d' % (i + 1): n for i, n in enumerate(nets)}, '200528-0100 (Molex)', note)


def pogo(ref, nets, note=''):
    n = len(nets)
    return Part(ref, 'Connector_Generic:Conn_01x%02d' % n, 'ayas:Pogo_1x%02d_P2.54mm' % n, 'POGO %d' % n,
                {'#%d' % (i + 1): x for i, x in enumerate(nets)}, 'Mill-Max 0906 series spring pins (verify)', note)


def pads(ref, nets, note=''):
    n = len(nets)
    return Part(ref, 'Connector_Generic:Conn_01x%02d' % n, 'ayas:PogoPads_1x%02d_P2.54mm' % n, 'PADS %d' % n,
                {'#%d' % (i + 1): x for i, x in enumerate(nets)}, 'ENIG pads', note)


def m2_holes(n, prefix='H'):
    return [Part('%s%d' % (prefix, i + 1), 'Mechanical:MountingHole', 'MountingHole:MountingHole_2.2mm_M2', 'M2', {})
            for i in range(n)]


def mount_holes(prefix='H'):
    return [Part('%s%d' % (prefix, i + 1), 'Mechanical:MountingHole_Pad', 'MountingHole:MountingHole_3.2mm_M3_DIN965_Pad',
                 'M3', {'#1': 'GND'}) for i in range(4)]


# ------------------------------------------------------------------------------------------------
# POWER board (right carrier, 46 x 55 mm): USB-C PD sink -> 12 V; 5.1 V / 5 A buck for the Pi;
# INA226 on the LED rail.  Interboard link J2 to LIGHT/IO.
# ------------------------------------------------------------------------------------------------
def power_board():
    p = []
    # USB-C receptacle from the power bank (right-angle C-C cable up through the shelf slot)
    p.append(Part('J1', 'Connector:USB_C_Receptacle_USB2.0_16P', 'Connector_USB:USB_C_Receptacle_GCT_USB4105-xx-A_16P_TopMnt_Horizontal',
                  'USB-C', {'VBUS': 'VBUS', 'GND': 'GND', 'SHIELD': 'GND', 'CC1': 'BANK_CC1', 'CC2': 'BANK_CC2'},
                  'GCT USB4105-GF-A', 'power + CC only; D+/D-/SBU not connected'))
    p.append(Part('D1', 'Device:D_TVS', 'Diode_SMD:D_SMB', 'SMBJ24A', {'#1': 'VBUS', '#2': 'GND'}, 'SMBJ24A (Littelfuse)',
                  'VBUS clamp; charging through the body port may put up to 20 V here'))
    # STUSB4500, DS12499 §6.2: VDD from VBUS; VSYS to GND; 1 uF on each regulator; dead-battery CCxDB -> CCx
    p.append(Part('U1', 'Interface_USB:STUSB4500QTR', 'Package_DFN_QFN:QFN-24-1EP_4x4mm_P0.5mm_EP2.6x2.6mm', 'STUSB4500',
                  {'CC1': 'PD_CC1', 'CC1DB': 'PD_CC1', 'CC2': 'PD_CC2', 'CC2DB': 'PD_CC2', 'VDD': 'VBUS', 'VSYS': 'GND',
                   'VREG_2V7': 'VREG_2V7', 'VREG_1V2': 'VREG_1V2', 'VBUS_VS_DISCH': 'VBUS_SNS', 'DISCH': 'DISCH',
                   'VBUS_EN_SNK': 'PD_GATE', 'ADDR0': 'GND', 'ADDR1': 'GND', 'RESET': 'PD_RESET',
                   'SCL': 'SCL', 'SDA': 'SDA', 'ALERT': 'ALERT_N', 'GND': 'GND'},
                  'STUSB4500QTR (ST)', 'I2C 0x28; NVM: PDO2 = 12 V 2.5 A, POWER_ONLY_ABOVE_5V = 1'))
    p += [C('C1', '1u', 'VBUS', 'GND'), C('C2', '1u', 'VREG_2V7', 'GND'), C('C3', '1u', 'VREG_1V2', 'GND'),
          R('R1', '1k', 'VBUS', 'VBUS_SNS'),              # VBUS_VS_DISCH series resistor (max 50 mA)
          R('R2', '1k', 'V12', 'DISCH', fp='Resistor_SMD:R_1206_3216Metric'),   # DISCH series resistor
          R('R3', '100k', 'VBUS', 'PD_GATE'),             # PMOS gate pull-up (switch off by default)
          R('R4', '10k', 'PD_RESET', 'GND')]              # RESET active high: held low
    # VBUS power-path switch (P-MOSFET), driven by VBUS_EN_SNK (HV open drain)
    p.append(Part('Q1', 'Transistor_FET:AO3401A', 'Package_TO_SOT_SMD:SOT-23', 'AO3401A',
                  {'G': 'PD_GATE', 'S': 'VBUS', 'D': 'V12_SW'}, 'AO3401A (AOS)', '-30 V, 4 A; ~2.2 A peak load'))
    p.append(Part('F1', 'Device:Polyfuse', 'Fuse:Fuse_1812_4532Metric', '2.6 A hold', {'#1': 'V12_SW', '#2': 'V12'},
                  'MF-MSMF260-2 (Bourns)'))
    p += [C('C4', '22u', 'V12', 'GND', 'Capacitor_SMD:C_1210_3225Metric', '25 V X5R'),
          C('C5', '22u', 'V12', 'GND', 'Capacitor_SMD:C_1210_3225Metric', '25 V X5R')]
    # 5.1 V / 5 A buck for the Pi (TPS565201, SLVSE71 Table 2 values for 5 V; R1 = 57.6k gives 5.14 V)
    p.append(Part('U2', 'Regulator_Switching:TPS565208', 'Package_TO_SOT_SMD:SOT-23-6', 'TPS565201',
                  {'VIN': 'V12', 'GND': 'GND', 'SW': 'SW', 'VBST': 'BST', 'EN': 'BUCK_EN', 'VFB': 'FB'},
                  'TPS565201DDCR (TI)', 'symbol TPS565208: same DDC pinout (GND SW VIN VFB EN VBST)'))
    p += [C('C6', '10u', 'V12', 'GND', 'Capacitor_SMD:C_1206_3216Metric', '25 V'),
          C('C7', '10u', 'V12', 'GND', 'Capacitor_SMD:C_1206_3216Metric', '25 V'),
          C('C8', '100n', 'V12', 'GND'),
          C('C9', '100n', 'BST', 'SW'),
          R('R5', '100k', 'V12', 'BUCK_EN'),
          R('R6', '57.6k', 'V5', 'FB'), R('R7', '10k', 'FB', 'GND'),
          C('C10', '22u', 'V5', 'GND', 'Capacitor_SMD:C_1206_3216Metric', '10 V X5R'),
          C('C11', '22u', 'V5', 'GND', 'Capacitor_SMD:C_1206_3216Metric', '10 V X5R'),
          C('C12', '22u', 'V5', 'GND', 'Capacitor_SMD:C_1206_3216Metric', '10 V X5R')]
    p.append(Part('L1', 'Device:L', 'Inductor_SMD:L_Bourns_SRP7028A_7.3x6.6mm', '3.3u', {'#1': 'SW', '#2': 'V5'},
                  'SRP7028A-3R3M (Bourns, 9 A sat)'))
    # INA226 on the 12 V LED rail (shunt R8), powered from the LIGHT board's 3.3 V so I2C levels match
    p.append(Part('R8', 'Device:R_Shunt', 'Resistor_SMD:R_Shunt_Vishay_WSK2512_6332Metric_T1.19mm', '20m',
                  {'#1': 'V12', '#2': 'V12_LED', '#3': 'SH_P', '#4': 'SH_N'}, 'WSK2512R0200FEA (Vishay)', 'Kelvin shunt'))
    p.append(Part('U3', 'Sensor_Energy:INA226', 'Package_SO:MSOP-10_3x3mm_P0.5mm', 'INA226',
                  {'Vin+': 'SH_P', 'Vin-': 'SH_N', 'Vbus': 'V12_LED', 'VS': '3V3', 'GND': 'GND', 'A0': 'GND', 'A1': 'GND',
                   'SCL': 'SCL', 'SDA': 'SDA', '~{Alert}': 'ALERT_N'}, 'INA226AIDGSR (TI)', 'I2C 0x40'))
    p.append(C('C13', '100n', '3V3', 'GND'))
    # ---- power button, power hold and charge-through (docs/hardware/pcb_phase2.md section 8) ----
    # Two small always-on 3.3 V rails (LM2936: 40 V in, ~15 uA quiescent):
    #   3V3_AON from the bank VBUS (button pull-up, button LED); 3V3_CHG from a plugged-in charger only.
    p += [Part('U4', 'Regulator_Linear:LM2936-3.3', 'Package_TO_SOT_SMD:SOT-223-3_TabPin2', 'LM2936-3.3',
               {'VI': 'VBUS', 'GND': 'GND', 'VO': '3V3_AON'}, 'LM2936MP-3.3 (TI)'),
          C('C14', '1u', 'VBUS', 'GND', 'Capacitor_SMD:C_0805_2012Metric', '50 V'),
          C('C15', '10u', '3V3_AON', 'GND', 'Capacitor_SMD:C_0805_2012Metric', 'LM2936 needs >= 10 uF out'),
          Part('U6', 'Regulator_Linear:LM2936-3.3', 'Package_TO_SOT_SMD:SOT-223-3_TabPin2', 'LM2936-3.3',
               {'VI': 'CHG_VBUS', 'GND': 'GND', 'VO': '3V3_CHG'}, 'LM2936MP-3.3 (TI)'),
          C('C17', '1u', 'CHG_VBUS', 'GND', 'Capacitor_SMD:C_0805_2012Metric', '50 V'),
          C('C18', '10u', '3V3_CHG', 'GND', 'Capacitor_SMD:C_0805_2012Metric'),
          # router supply = DEVICE_MODE OR 3V3_CHG: with neither, the switch is unpowered and conducts nothing,
          # so after shutdown the bank sees a clean detach (no Rd anywhere) and turns its output off
          Part('D2', 'Diode:BAT54C', 'Package_TO_SOT_SMD:SOT-23', 'BAT54C', {'#1': 'DEVICE_MODE', '#2': '3V3_CHG', '#3': 'VPLUS'},
               'BAT54C (Nexperia)', 'router supply OR')]
    # Power button (momentary, ring LED): pressing grounds BTN; Rd 5.1k via Schottkys on both bank CC lines
    # wakes the bank (a USB-C source only powers VBUS when it sees Rd), and Q2 forces DEVICE_MODE on.
    p += [R('R9', '5.1k', 'BANK_CC1', 'WAKE1'), R('R10', '5.1k', 'BANK_CC2', 'WAKE2'),
          Part('D3', 'Diode:BAT54C', 'Package_TO_SOT_SMD:SOT-23', 'BAT54C', {'#1': 'WAKE1', '#2': 'WAKE2', '#3': 'BTN_N'},
               'BAT54C (Nexperia)', 'isolates CC1 from CC2 when the button is up'),
          R('R11', '100k', '3V3_AON', 'BTN_N'),
          Part('Q2', 'Transistor_FET:BSS84', 'Package_TO_SOT_SMD:SOT-23', 'BSS84', {'G': 'BTN_N', 'S': '3V3_AON', 'D': 'BTN_ON'},
               'BSS84 (Nexperia)'),
          Part('D4', 'Diode:BAT54C', 'Package_TO_SOT_SMD:SOT-23', 'BAT54C', {'#1': 'BTN_ON', '#2': 'HOLD', '#3': 'DEVICE_MODE'},
               'BAT54C (Nexperia)', 'DEVICE_MODE = button OR HOLD (RP2040)'),
          R('R12', '100k', 'DEVICE_MODE', 'GND'), R('R13', '100k', 'HOLD', 'GND')]
    # CC router: DEVICE_MODE high -> bank CC to the STUSB4500 (NO); low or unpowered side -> bank CC to the
    # body charge port (NC), so the bank sees "detached" (output off) or charges from the port.
    p.append(Part('U5', 'Analog_Switch:TS5A23157DGS', 'Package_SO:TSSOP-10_3x3mm_P0.5mm', 'TS5A23157',
                  {'#1': 'DEVICE_MODE', '#2': 'PD_CC1', '#3': 'GND', '#4': 'PD_CC2', '#5': 'DEVICE_MODE', '#6': 'BANK_CC2',
                   '#7': 'CHG_CC2', '#8': 'VPLUS', '#9': 'CHG_CC1', '#10': 'BANK_CC1'}, 'TS5A23157DGSR (TI)',
                  'IN low: COM-NC; IN high: COM-NO (SCDS165F pin table)'))
    p.append(C('C16', '100n', 'VPLUS', 'GND'))
    # charge port CC: Rd so a charger supplies 5 V at once (then the bank negotiates through the router)
    p += [R('R14', '5.1k', 'CHG_CC1', 'GND'), R('R15', '5.1k', 'CHG_CC2', 'GND')]
    # charge VBUS path: back-to-back P-FETs, on only when a charger is present AND the device is off
    p += [Part('Q3', 'Transistor_FET:AO3401A', 'Package_TO_SOT_SMD:SOT-23', 'AO3401A', {'G': 'CHG_G', 'S': 'CHG_MID', 'D': 'CHG_VBUS'},
               'AO3401A (AOS)'),
          Part('Q4', 'Transistor_FET:AO3401A', 'Package_TO_SOT_SMD:SOT-23', 'AO3401A', {'G': 'CHG_G', 'S': 'CHG_MID', 'D': 'VBUS'},
               'AO3401A (AOS)'),
          R('R16', '100k', 'CHG_MID', 'CHG_G'),
          Part('D5', 'Diode:BZT52Bxx', 'Diode_SMD:D_SOD-123F', 'BZT52B10', {'K': 'CHG_MID', 'A': 'CHG_G'}, 'BZT52B10 (Nexperia)',
               'limits Vgs to -10 V (AO3401A max 12 V) at a 20 V charge contract'),
          R('R17', '10k', 'CHG_G', 'CHG_PULL'),
          Part('Q5', 'Transistor_FET:2N7002', 'Package_TO_SOT_SMD:SOT-23', '2N7002', {'G': 'CHG_ON', 'S': 'GND', 'D': 'CHG_PULL'},
               '2N7002 (Nexperia)'),
          R('R18', '100k', 'CHG_VBUS', 'CHG_ON'), R('R19', '100k', 'CHG_ON', 'GND'),
          Part('Q6', 'Transistor_FET:2N7002', 'Package_TO_SOT_SMD:SOT-23', '2N7002', {'G': 'DEVICE_MODE', 'S': 'GND', 'D': 'CHG_ON'},
               '2N7002 (Nexperia)', 'no charge-through while the device is on')]
    # power-button ring LED: lit while DEVICE_MODE
    p += [R('R20', '330', '3V3_AON', 'BTN_LED_A'),
          Part('Q7', 'Transistor_FET:2N7002', 'Package_TO_SOT_SMD:SOT-23', '2N7002', {'G': 'DEVICE_MODE', 'S': 'GND', 'D': 'BTN_LED_K'},
               '2N7002 (Nexperia)')]
    p.append(Part('J5', 'Connector_Generic:Conn_01x04', 'Connector_JST:JST_PH_B4B-PH-K_1x04_P2.00mm_Vertical', 'PWR BTN',
                  {'#1': 'BTN_N', '#2': 'GND', '#3': 'BTN_LED_A', '#4': 'BTN_LED_K'}, 'B4B-PH-K-S (JST)',
                  '12 mm momentary button with ring LED, top of the body'))
    p.append(jst_ph('J6', ['CHG_VBUS', 'CHG_CC1', 'CHG_CC2', 'GND', 'GND'], 'CHARGE: charge-port board (PH-5; PWR BTN is PH-4)'))
    # outputs
    p.append(Part('J2', 'Connector_Generic:Conn_01x12', 'Connector_JST:JST_PH_B12B-PH-K_1x12_P2.00mm_Vertical',
                  'LINK', LINK, 'B12B-PH-K-S (JST)', 'to LIGHT/IO J1'))
    p.append(Part('J3', 'Connector_Generic:Conn_01x02', 'Connector_JST:JST_VH_B2P-VH_1x02_P3.96mm_Vertical',
                  'PI 5V', {'#1': 'V5', '#2': 'GND'}, 'B2P-VH (JST)', 'to the Pi 5 USB-C input (pigtail), 5 A'))
    p += mount_holes()
    return p


# ------------------------------------------------------------------------------------------------
# LIGHT/IO board (left carrier, 46 x 55 mm): RP2040 on USB to the Pi; 10 constant-current LED sinks;
# buttons, hood loop-back + UV interlock, fan, buzzer, NTCs.
# ------------------------------------------------------------------------------------------------
CH = [('WHITE_L', 1.5), ('WHITE_R', 1.5), ('UV_L', 3.3), ('UV_R', 3.3),
      ('RING1', 3.3), ('RING2', 3.3), ('RING3', 3.3), ('RING4', 3.3), ('RING5', 3.3), ('RING6', 3.3)]
# set-point: RP2040 GPIO (3.3 V) / (150k + 10k) -> 0.206 V full scale; I = 0.206 / Rsense:
#   white 137 mA per pod (3 x 12 V white modules in parallel on the WHITE carrier),
#   UV 62 mA (3 strings x 20 mA on the plug-in UV board), ring 62 mA (one white module per channel).
# The modules have their own resistors, so the firmware sets each channel ~80 % of the module's own current
# at 12 V (measure it once on the bench): the sink then regulates and the light stays the same burst to burst.


def light_board():
    p = []
    p.append(Part('J1', 'Connector_Generic:Conn_01x12', 'Connector_JST:JST_PH_B12B-PH-K_1x12_P2.00mm_Vertical',
                  'LINK', LINK, 'B12B-PH-K-S (JST)', 'from POWER J2'))
    # 3.3 V LDO from 5 V
    p.append(Part('U1', 'Regulator_Linear:AP2112K-3.3', 'Package_TO_SOT_SMD:SOT-23-5', 'AP2112K-3.3',
                  {'VIN': 'V5', 'EN': 'V5', 'GND': 'GND', 'VOUT': '3V3'}, 'AP2112K-3.3TRG1 (Diodes)'))
    p += [C('C1', '1u', 'V5', 'GND'), C('C2', '1u', '3V3', 'GND'), C('C3', '10u', '3V3', 'GND', 'Capacitor_SMD:C_0805_2012Metric')]
    # RP2040 (Raspberry Pi RP2040 hardware design guide: 12 MHz crystal, W25Q16 flash, 1.1 V core from VREG)
    gp = {}
    for i, (name, _) in enumerate(CH):
        gp['GPIO%d' % i] = 'EN_' + name
    for i, b in enumerate(['TRIG_L', 'TRIG_R', 'THUMB1', 'THUMB2', 'THUMB3', 'THUMB4']):
        gp['GPIO%d' % (10 + i)] = b
    gp.update({'GPIO22': 'PI_ON', 'GPIO23': 'BTN_N', 'GPIO24': 'HOLD'})     # power: Pi USB sense, power button, hold
    gp.update({'GPIO16': 'HOOD_N', 'GPIO17': 'FAN_PWM', 'GPIO18': 'BUZ', 'GPIO19': 'ALERT_N',
               'GPIO20': 'SDA', 'GPIO21': 'SCL', 'GPIO26/ADC0': 'NTC_L', 'GPIO27/ADC1': 'NTC_R', 'GPIO28/ADC2': 'NTC_B'})
    p.append(Part('U2', 'MCU_RaspberryPi:RP2040', 'Package_DFN_QFN:QFN-56-1EP_7x7mm_P0.4mm_EP3.2x3.2mm', 'RP2040',
                  dict(gp, **{'IOVDD': '3V3', 'USB_VDD': '3V3', 'ADC_AVDD': '3V3', 'VREG_VIN': '3V3', 'VREG_VOUT': '1V1',
                              'DVDD': '1V1', 'GND': 'GND', 'TESTEN': 'GND', 'XIN': 'XIN', 'XOUT': 'XOUT_R',
                              'RUN': 'RUN', 'USB_DP': 'USBP_C', 'USB_DM': 'USBM_C', '~{QSPI_SS}': 'QSPI_SS',
                              'QSPI_SCLK': 'QSPI_SCLK', 'QSPI_SD0': 'QSPI_SD0', 'QSPI_SD1': 'QSPI_SD1',
                              'QSPI_SD2': 'QSPI_SD2', 'QSPI_SD3': 'QSPI_SD3', 'SWCLK': 'SWCLK', 'SWDIO': 'SWDIO'}),
                  'RP2040 (Raspberry Pi)'))
    p += [C('C%d' % (4 + i), '100n', '3V3', 'GND', 'Capacitor_SMD:C_0402_1005Metric') for i in range(6)]
    p += [C('C10', '1u', '3V3', 'GND'), C('C11', '1u', '1V1', 'GND'), C('C12', '100n', '1V1', 'GND', 'Capacitor_SMD:C_0402_1005Metric'),
          C('C13', '100n', '1V1', 'GND', 'Capacitor_SMD:C_0402_1005Metric')]
    p.append(Part('U3', 'Memory_Flash:W25Q16JVSS', 'Package_SO:SOIC-8_5.3x5.3mm_P1.27mm', 'W25Q16JV',
                  {'~{CS}': 'QSPI_SS', 'DO/IO_{1}': 'QSPI_SD1', '~{WP}/IO_{2}': 'QSPI_SD2', 'GND': 'GND', 'DI/IO_{0}': 'QSPI_SD0',
                   'CLK': 'QSPI_SCLK', '~{HOLD}/~{RESET}/IO_{3}': 'QSPI_SD3', 'VCC': '3V3'}, 'W25Q16JVSSIQ (Winbond)'))
    p += [C('C14', '100n', '3V3', 'GND'), R('R1', '10k', '3V3', 'QSPI_SS')]
    p.append(Part('Y1', 'Device:Crystal_GND24', 'Crystal:Crystal_SMD_3225-4Pin_3.2x2.5mm', '12MHz',
                  {'#1': 'XIN', '#3': 'XOUT', '#2': 'GND', '#4': 'GND'}, 'ABM8-272-T3 (Abracon)'))
    p += [C('C15', '15p', 'XIN', 'GND'), C('C16', '15p', 'XOUT', 'GND'), R('R2', '1k', 'XOUT_R', 'XOUT')]
    p += [R('R3', '10k', '3V3', 'RUN')]
    p.append(Part('SW1', 'Switch:SW_Push', 'Button_Switch_SMD:SW_SPST_B3U-1000P', 'BOOT', {'#1': 'QSPI_SS_B', '#2': 'GND'},
                  'B3U-1000P (Omron)'))
    p.append(R('R4', '1k', 'QSPI_SS', 'QSPI_SS_B'))
    p.append(Part('SW2', 'Switch:SW_Push', 'Button_Switch_SMD:SW_SPST_B3U-1000P', 'RESET', {'#1': 'RUN', '#2': 'GND'},
                  'B3U-1000P (Omron)'))
    # USB to the Pi (JST-PH 4 -> USB-A pigtail); 27 ohm series per the RP2040 design guide
    p += [R('R5', '27', 'USBP_C', 'USB_DP'), R('R6', '27', 'USBM_C', 'USB_DM')]
    p.append(jst_ph('J2', ['VBUS_PI', 'USB_DM', 'USB_DP', 'GND'], 'USB: to a Pi 5 USB 2.0 port (PH-4, cannot take the SH-4 SIDE-L cable); VBUS_PI only sensed'))
    p += [R('R7', '10k', 'VBUS_PI', 'PI_ON'), R('R8', '20k', 'PI_ON', 'GND')]    # Pi USB 5 V -> 3.3 V: Pi has shut down when low
    # constant-current sinks: LM324 x 3 (12 amps, 10 used) + AO3400A + sense resistor
    amps = [('U4', 'A'), ('U4', 'B'), ('U4', 'C'), ('U4', 'D'), ('U5', 'A'), ('U5', 'B'), ('U5', 'C'), ('U5', 'D'),
            ('U6', 'A'), ('U6', 'B'), ('U6', 'C'), ('U6', 'D')]
    unit_pins = {'A': (3, 2, 1), 'B': (5, 6, 7), 'C': (10, 9, 8), 'D': (12, 13, 14)}   # (+in, -in, out) LM324 SOIC-14
    opamp_pins = {u: {'#4': 'V5', '#11': 'GND'} for u in ('U4', 'U5', 'U6')}
    r, c, q = 10, 17, 1
    for i, (name, rs) in enumerate(CH):
        u, unit = amps[i]
        pin_p, pin_n, pin_o = unit_pins[unit]
        sp, fb, gate, src = 'SP_' + name, 'FB_' + name, 'G_' + name, 'S_' + name
        opamp_pins[u].update({'#%d' % pin_p: sp, '#%d' % pin_n: fb, '#%d' % pin_o: 'OA_' + name})
        p += [R('R%d' % r, '150k', 'EN_' + name, sp), R('R%d' % (r + 1), '10k', sp, 'GND'),
              R('R%d' % (r + 2), '100', 'OA_' + name, gate), R('R%d' % (r + 3), '1k', src, fb),
              C('C%d' % c, '10n', 'OA_' + name, fb),
              R('R%d' % (r + 4), str(rs), src, 'GND', 'Resistor_SMD:R_1206_3216Metric')]
        p.append(Part('Q%d' % q, 'Transistor_FET:AO3400A', 'Package_TO_SOT_SMD:SOT-23', 'AO3400A',
                      {'G': gate, 'S': src, 'D': 'K_' + name}, 'AO3400A (AOS)'))
        r, c, q = r + 5, c + 1, q + 1
    for i, (u, unit) in enumerate(amps[len(CH):]):                     # spare amps: follower to GND
        pin_p, pin_n, pin_o = unit_pins[unit]
        opamp_pins[u].update({'#%d' % pin_p: 'GND', '#%d' % pin_n: 'SPARE%d' % i, '#%d' % pin_o: 'SPARE%d' % i})
    for u in ('U4', 'U5', 'U6'):
        p.append(Part(u, 'Amplifier_Operational:LM324', 'Package_SO:SOIC-14_3.9x8.7mm_P1.27mm', 'LM324',
                      opamp_pins[u], 'LM324DR (TI)'))
        p.append(C('C%d' % c, '100n', 'V5', 'GND')); c += 1
    # UV interlock: hood absent -> HOOD_N pulled high -> FETs clamp the UV set-points to GND
    p.append(R('R%d' % r, '10k', '3V3', 'HOOD_N')); r += 1
    for name in ('UV_L', 'UV_R'):
        p.append(Part('Q%d' % q, 'Transistor_FET:2N7002', 'Package_TO_SOT_SMD:SOT-23', '2N7002',
                      {'G': 'HOOD_N', 'S': 'GND', 'D': 'SP_' + name}, '2N7002 (Nexperia)')); q += 1
    # fan and buzzer (low side, flyback diodes)
    for name, net in (('FAN', 'FAN_PWM'), ('BUZ', 'BUZ')):
        p.append(Part('Q%d' % q, 'Transistor_FET:AO3400A', 'Package_TO_SOT_SMD:SOT-23', 'AO3400A',
                      {'G': net, 'S': 'GND', 'D': name + '_K'}, 'AO3400A (AOS)')); q += 1
        p.append(Part('D%d' % (1 if name == 'FAN' else 2), 'Device:D_Schottky', 'Diode_SMD:D_SOD-123', 'B0530W',
                      {'K': 'V5', 'A': name + '_K'}, 'B0530W (Diodes)'))
        p.append(R('R%d' % r, '100k', net, 'GND')); r += 1
    # NTCs: 10k NTC to GND, 10k bias from 3.3 V (pods carry their own NTC)
    for net in ('NTC_L', 'NTC_R', 'NTC_B'):
        p.append(R('R%d' % r, '10k', '3V3', net)); r += 1
        p.append(C('C%d' % c, '100n', net, 'GND')); c += 1
    p.append(Part('TH1', 'Device:Thermistor_NTC', 'Resistor_SMD:R_0603_1608Metric', '10k NTC', {'#1': 'NTC_B', '#2': 'GND'},
                  'NCP18XH103F03RB (Murata)', 'sink area'))
    # button inputs: series 1k + 100n (RP2040 internal pull-ups)
    btn = ['TRIG_L', 'TRIG_R', 'THUMB1', 'THUMB2', 'THUMB3', 'THUMB4']
    for b in btn:
        p.append(R('R%d' % r, '1k', b + '_IN', b)); r += 1
        p.append(C('C%d' % c, '100n', b, 'GND')); c += 1
    # connectors
    # one keyed connector per module; no two connectors on this board share family + pin count
    p.append(jst_sh('J3', POD_PINS('L') + ['GND'], 'POD-L: left pod dock (6-way)'))
    p.append(jst_sh('J4', POD_PINS('R') + ['GND', 'GND'], 'POD-R: right pod dock (7-way)'))
    p.append(jst_sh('J5', ['V12_LED', 'K_RING1', 'K_RING2', 'K_RING3', 'K_RING4', 'K_RING5', 'K_RING6', 'HOOD_N', 'GND'],
                    'HOOD: hood dock (9-way); the hood shorts HOOD_N to GND'))
    p.append(jst_sh('J6', ['TRIG_L_IN', 'THUMB1_IN', 'THUMB2_IN', 'GND'], 'SIDE-L: left wing, trigger + 2 thumbs (4-way)'))
    p.append(jst_sh('J7', ['TRIG_R_IN', 'THUMB3_IN', 'THUMB4_IN', 'GND', 'GND'], 'SIDE-R: right wing, trigger + 2 thumbs (5-way)'))
    p.append(jst_sh('J9', ['V5', 'FAN_K'], 'FAN (2-way)'))
    p.append(Part('BZ1', 'Device:Buzzer', 'Buzzer_Beeper:MagneticBuzzer_CUI_CMT-8504-100-SMT', 'CMT-8504',
                  {'+': 'V5', '-': 'BUZ_K'}, 'CMT-8504-100-SMT (CUI)', 'on board: no buzzer cable'))
    for i, net in enumerate(['3V3', 'SWDIO', 'SWCLK', 'GND']):          # SWD on flat pads (no height in the low strip)
        p.append(Part('TP%d' % (i + 1), 'Connector:TestPoint', 'TestPoint:TestPoint_Pad_D1.5mm', net, {'#1': net}))
    p.append(Part('C%d' % c, 'Device:C_Polarized', 'Capacitor_SMD:CP_Elec_6.3x7.7', '100u', {'#1': 'V12_LED', '#2': 'GND'},
                  '25 V', 'LED rail bulk')); c += 1
    p += mount_holes()
    return p


# ------------------------------------------------------------------------------------------------
# UV socket board, one per pod (43 x 30.5 mm, in the pod's UV cavity behind the tilted platform).
# 9 plug-in 5 mm UV LEDs (395 nm, owned stock) in 3 columns; each column is a string of 3 with its own
# ballast resistor: 12 V - 3 x 3.2 V - 47R x 20 mA - 0.2 V sense leaves ~1.2 V across the sink FET.
# Each LED has a 2-pin socket: pin 1 (square pad, "+") anode, pin 2 cathode. A flex tail (uv_flex)
# links J1 to J12/J13 on LIGHT/IO.
# ------------------------------------------------------------------------------------------------
UV_COLS = 3            # 3 columns x 3 LEDs = the 9 slots in the pod CAD (set 2 for 6 LEDs)
UV_ROWS = 3


def uv_board():
    """POD HUB: the UV socket board; its flex goes to the body's pod dock, and J2 feeds the pod's WHITE carrier."""
    p = [ffc10('J1', POD_FLEX, 'flex to the pod dock in the body front face'),
         jst_ph('J2', ['V12_LED', 'K_WHITE'], 'WHITE: to the WHITE-L/R carrier J4')]
    n = 1
    for c in range(UV_COLS):
        prev = 'V12_LED'
        for r in range(UV_ROWS):
            nxt = 'S%d_%d' % (c + 1, r + 1)
            p.append(Part('X%d' % n, 'Connector_Generic:Conn_01x02', 'Connector_PinSocket_2.54mm:PinSocket_1x02_P2.54mm_Vertical',
                          'UV LED', {'#1': prev, '#2': nxt}, '2-pin machined socket, 2.54 mm (Mill-Max 310-43-102-41-001000)',
                          'pin 1 = anode (+), square pad; 5 mm 395 nm UV LED plugs in'))
            prev, n = nxt, n + 1
        p.append(R('R%d' % (c + 1), '47', prev, 'K_UV', 'Resistor_SMD:R_0805_2012Metric'))
    p.append(Part('TH1', 'Device:Thermistor_NTC', 'Resistor_SMD:R_0603_1608Metric', '10k NTC', {'#1': 'NTC', '#2': 'GND'},
                  'NCP18XH103F03RB (Murata)', 'LED temperature'))
    p.append(C('C1', '1u', 'V12_LED', 'GND', 'Capacitor_SMD:C_0805_2012Metric', '25 V'))
    p += [Part('H%d' % (i + 1), 'Mechanical:MountingHole', 'MountingHole:MountingHole_2.2mm_M2', 'M2', {}) for i in range(2)]
    return p


def uv_flex(length):
    """Flex tail: two finger ends, eight straight conductors (palindrome pinout)."""
    def f():
        # eight straight conductors C1..C8; the far end is turned round, so P2 pin j carries C(9 - j).
        # The mating boards use the palindrome uv_pins(), so either flex orientation connects correctly.
        return [Part('P1', 'Connector_Generic:Conn_01x10', 'ayas:FFC_Fingers_1x10_P1.00mm', 'FFC 10',
                     {'#%d' % i: 'C%d' % i for i in range(1, 11)}, '', 'flex contact fingers, end 1'),
                Part('P2', 'Connector_Generic:Conn_01x10', 'ayas:FFC_Fingers_1x10_P1.00mm', 'FFC 10',
                     {'#%d' % j: 'C%d' % (11 - j) for j in range(1, 11)}, '', 'flex contact fingers, end 2')]
    f.length = length
    return f


# ------------------------------------------------------------------------------------------------
# CHARGE-PORT board (14 x 16 mm, in the bay corner behind the right side wall): USB-C power-only
# receptacle through the wall, cable to POWER J6. Charges the bank while the device is off.
# ------------------------------------------------------------------------------------------------
def charge_board():
    return [Part('J1', 'Connector:USB_C_Receptacle_PowerOnly_6P', 'Connector_USB:USB_C_Receptacle_GCT_USB4125-xx-x_6P_TopMnt_Horizontal',
                 'USB-C', {'VBUS': 'CHG_VBUS', 'GND': 'GND', 'SHIELD': 'GND', 'CC1': 'CHG_CC1', 'CC2': 'CHG_CC2'},
                 'USB4125-GF-A (GCT)', 'power-only USB-C, through the side wall'),
            Part('D1', 'Device:D_TVS', 'Diode_SMD:D_SMA', 'SMAJ24A', {'#1': 'CHG_VBUS', '#2': 'GND'}, 'SMAJ24A (Littelfuse)'),
            jst_ph('J2', ['CHG_VBUS', 'CHG_CC1', 'CHG_CC2', 'GND', 'GND'], 'CHARGE: to POWER J6'),
            Part('H1', 'Mechanical:MountingHole', 'MountingHole:MountingHole_2.2mm_M2', 'M2', {})]


# ------------------------------------------------------------------------------------------------
# WHITE light: the off-the-shelf 12 V white LED module (3 x 2835 SMD LEDs + its own resistor, 40 x 10 mm
# housing with a screw tab at each end; the owned stock is marked 3609-2835-3LED). Used 6x in the pods (3 per
# side, on the WHITE-L and WHITE-R carrier boards) and 6x round the hood ring (on the hood pad board).
# Each module: 2 x M2 x 6 pan screws through its tabs into SMD nuts on the carrier's back, and its two leads
# (cut from the daisy chain, stripped 5 mm) in a push-in terminal: no solder, no crimp tool, no adhesive.
# ------------------------------------------------------------------------------------------------
WHITE_MOD = dict(L=40.0, W=10.0, hole_pitch=46.0, pitch=14.0)   # hole_pitch: tab hole centres, VERIFY with calipers


def push2(ref, nets, note=''):
    """Push-in terminal for a module's two leads (0.14-0.5 mm2, strip 5 mm; press the button to release)."""
    return Part(ref, 'Connector_Generic:Conn_01x02',
                'TerminalBlock_Phoenix:TerminalBlock_Phoenix_PTSM-0,5-2-2.5-H-THR_1x02_P2.50mm_Horizontal', 'PTSM 2',
                {'#1': nets[0], '#2': nets[1]}, 'PTSM 0,5/ 2-2,5-H THR (Phoenix Contact)', note)


def smd_nut(ref, note='module tab screw (M2 x 6)'):
    return Part(ref, 'Mechanical:MountingHole', 'Mounting_Wuerth:Mounting_Wuerth_WA-SMSI-M2_H2mm_9774020243',
                'M2 nut', {}, '9774020243 (Wurth WA-SMSI, M2, 2 mm)', note)


def white_board(side):
    """WHITE-L / WHITE-R carrier: 3 modules in parallel on the pod's white channel; inner edge under a lip, 2 x M2 at the outer edge."""
    def f():
        p = [push2('J%d' % (i + 1), ['V12_LED', 'K_WHITE'], 'M%d: white module %d (+ red, - black)' % (i + 1, i + 1))
             for i in range(3)]
        p.append(Part('J4', 'Connector_Generic:Conn_01x02', 'Connector_JST:JST_PH_S2B-PH-K_1x02_P2.00mm_Horizontal', 'JST-PH 2',
                      {'#1': 'V12_LED', '#2': 'K_WHITE'}, 'S2B-PH-K-S (JST)',
                      'WHITE-%s: to the pod hub J2; side entry, plug toward the hub (the pod is shallow here)' % side))
        p += [smd_nut('N%d' % (i + 1)) for i in range(6)]
        return p + m2_holes(2)
    return f


# BUTTON boards, one in each wing (SIDE-L, SIDE-R): the panel buttons' two leads go into push-in terminals
# (switches have no polarity), one keyed JST-SH cable to LIGHT/IO (SIDE-L 4-way J6, SIDE-R 5-way J7).
# The debounce (1k + 100n) and pull-ups are on LIGHT/IO, so these boards are passive. 2 x M2 into the wing.
BTN_SIDE = {'L': (['TRIG_L', 'THUMB1', 'THUMB2'], 4, 6), 'R': (['TRIG_R', 'THUMB3', 'THUMB4'], 5, 7)}
BTN_LABEL = ['TRIG', 'UPPER', 'LOWER']        # index trigger, thumb upper, thumb lower


def button_board(side):
    def f():
        sig, n, jl = BTN_SIDE[side]
        p = [push2('J%d' % (i + 1), [x + '_IN', 'GND'], '%s button (%s)' % (BTN_LABEL[i], x)) for i, x in enumerate(sig)]
        p.append(jst_sh('J4', [x + '_IN' for x in sig] + ['GND'] * (n - 3), 'SIDE-%s: to LIGHT/IO J%d (%d-way)' % (side, jl, n)))
        return p + m2_holes(2)
    return f


# POD DOCK: screwed to the body front face behind each pod (2 x M2 inserts). The pod's flex plugs into its ZIF;
# a keyed cable goes through the wall to LIGHT/IO (POD-L 6-way, POD-R 7-way, so they cannot be swapped).
# The dock, not the pod, decides the side: the two pod modules are electrically identical.
def pod_dock_board(side):
    def f():
        n = 6 if side == 'L' else 7
        nets = POD_PINS(side) + ['GND'] * (n - 5)
        flex = ['GND', 'V12_LED', 'K_UV_' + side, 'K_WHITE_' + side, 'NTC_' + side, 'NTC_' + side,
                'K_WHITE_' + side, 'K_UV_' + side, 'V12_LED', 'GND']
        return [ffc10('J1', flex, 'pod flex plugs in here (palindrome: either way round)'),
                jst_sh('J2', nets, 'POD-%s: to LIGHT/IO J%d (%d-way)' % (side, 3 if side == 'L' else 4, n))] + m2_holes(2)
    return f


# HOOD PAD board in the collar top: 6 ring-module leads in (push-in), 9 contacts out; HOOD_N looped to GND here.
def hood_pad_board():
    p = [pads('J1', HOOD_PADS, 'contacts to the body hood dock (face down)'),
         R('R1', '0', 'HOOD_N', 'GND')]
    for i in range(6):
        p.append(push2('J%d' % (2 + i), ['V12_LED', 'K_RING%d' % (i + 1)], 'R%d: hood ring module %d leads' % (i + 1, i + 1)))
    return p + m2_holes(2)


def hood_dock_board():
    return [pogo('J1', HOOD_PADS, 'spring pins up through the top wall'),
            jst_sh('J2', ['V12_LED', 'K_RING1', 'K_RING2', 'K_RING3', 'K_RING4', 'K_RING5', 'K_RING6', 'HOOD_N', 'GND'],
                   'HOOD: to LIGHT/IO J5 (9-way)')] + m2_holes(2)


BOARDS = {'power': power_board, 'charge': charge_board, 'light': light_board, 'uv': uv_board,
          'podflex': uv_flex(60.0), 'white_l': white_board('L'), 'white_r': white_board('R'),
          'btn_l': button_board('L'), 'btn_r': button_board('R'),
          'pod_dock_l': pod_dock_board('L'), 'pod_dock_r': pod_dock_board('R'),
          'hood_pad': hood_pad_board, 'hood_dock': hood_dock_board}
