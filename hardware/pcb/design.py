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
                  'USB-C', {'VBUS': 'VBUS', 'GND': 'GND', 'SHIELD': 'GND', 'CC1': 'CC1', 'CC2': 'CC2'},
                  'GCT USB4105-GF-A', 'power + CC only; D+/D-/SBU not connected'))
    p.append(Part('D1', 'Device:D_TVS', 'Diode_SMD:D_SMB', 'SMBJ20A', {'#1': 'VBUS', '#2': 'GND'}, 'SMBJ20A (Littelfuse)',
                  'VBUS clamp (PD contract max 15 V)'))
    # STUSB4500, DS12499 §6.2: VDD from VBUS; VSYS to GND; 1 uF on each regulator; dead-battery CCxDB -> CCx
    p.append(Part('U1', 'Interface_USB:STUSB4500QTR', 'Package_DFN_QFN:QFN-24-1EP_4x4mm_P0.5mm_EP2.6x2.6mm', 'STUSB4500',
                  {'CC1': 'CC1', 'CC1DB': 'CC1', 'CC2': 'CC2', 'CC2DB': 'CC2', 'VDD': 'VBUS', 'VSYS': 'GND',
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
    # outputs
    p.append(Part('J2', 'Connector_Generic:Conn_01x10', 'Connector_JST:JST_PH_B10B-PH-K_1x10_P2.00mm_Vertical',
                  'LINK', {'#1': 'V12_LED', '#2': 'V12_LED', '#3': 'GND', '#4': 'GND', '#5': 'V5', '#6': '3V3',
                           '#7': 'SDA', '#8': 'SCL', '#9': 'ALERT_N', '#10': 'GND'}, 'B10B-PH-K-S (JST)', 'to LIGHT/IO J1'))
    p.append(Part('J3', 'Connector_Generic:Conn_01x02', 'Connector_JST:JST_VH_B2P-VH_1x02_P3.96mm_Vertical',
                  'PI 5V', {'#1': 'V5', '#2': 'GND'}, 'B2P-VH (JST)', 'to the Pi 5 USB-C input (pigtail), 5 A'))
    p += mount_holes()
    return p


# ------------------------------------------------------------------------------------------------
# LIGHT/IO board (left carrier, 46 x 55 mm): RP2040 on USB to the Pi; 10 constant-current LED sinks;
# buttons, hood loop-back + UV interlock, fan, buzzer, NTCs.
# ------------------------------------------------------------------------------------------------
CH = [('WHITE_L', 0.56), ('WHITE_R', 0.56), ('UV_L', 0.82), ('UV_R', 0.82),
      ('RING1', 1.0), ('RING2', 1.0), ('RING3', 1.0), ('RING4', 1.0), ('RING5', 1.0), ('RING6', 1.0)]
# set-point: RP2040 GPIO (3.3 V) / (150k + 10k) -> 0.206 V; I = 0.206 / Rsense: 368 mA, 251 mA, 206 mA


def light_board():
    p = []
    p.append(Part('J1', 'Connector_Generic:Conn_01x10', 'Connector_JST:JST_PH_B10B-PH-K_1x10_P2.00mm_Vertical',
                  'LINK', {'#1': 'V12_LED', '#2': 'V12_LED', '#3': 'GND', '#4': 'GND', '#5': 'V5', '#6': '3V3',
                           '#7': 'SDA', '#8': 'SCL', '#9': 'ALERT_N', '#10': 'GND'}, 'B10B-PH-K-S (JST)', 'from POWER J2'))
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
    p.append(jst_sh('J2', ['VBUS_PI', 'USB_DM', 'USB_DP', 'GND'], 'to a Pi 5 USB 2.0 port; VBUS_PI only sensed'))
    p.append(R('R7', '10k', 'VBUS_PI', 'GND'))
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
    p.append(jst_sh('J3', ['V12_LED', 'K_WHITE_L', 'K_UV_L', 'NTC_L', 'GND'], 'left pod'))
    p.append(jst_sh('J4', ['V12_LED', 'K_WHITE_R', 'K_UV_R', 'NTC_R', 'GND'], 'right pod'))
    p.append(jst_sh('J5', ['V12_LED', 'K_RING1', 'K_RING2', 'K_RING3', 'K_RING4', 'K_RING5', 'K_RING6', 'HOOD_N', 'GND'],
                    'hood ring harness; the hood shorts HOOD_N to GND'))
    p.append(jst_sh('J6', ['TRIG_L_IN', 'GND'], 'left trigger'))
    p.append(jst_sh('J7', ['TRIG_R_IN', 'GND'], 'right trigger'))
    p.append(jst_sh('J8', ['THUMB1_IN', 'THUMB2_IN', 'THUMB3_IN', 'THUMB4_IN', 'GND'], 'thumb buttons'))
    p.append(jst_sh('J9', ['V5', 'FAN_K'], '30 mm fan'))
    p.append(jst_sh('J10', ['V5', 'BUZ_K'], 'buzzer'))
    for i, net in enumerate(['3V3', 'SWDIO', 'SWCLK', 'GND']):          # SWD on flat pads (no height in the low strip)
        p.append(Part('TP%d' % (i + 1), 'Connector:TestPoint', 'TestPoint:TestPoint_Pad_D1.5mm', net, {'#1': net}))
    p.append(Part('C%d' % c, 'Device:C_Polarized', 'Capacitor_SMD:CP_Elec_6.3x7.7', '100u', {'#1': 'V12_LED', '#2': 'GND'},
                  '25 V', 'LED rail bulk')); c += 1
    p += mount_holes()
    return p


BOARDS = {'power': power_board, 'light': light_board}
