"""
Connect two Voxel devices to WiFi using different approaches.

Methods tried:
 - serial_serial: open two serial ports and issue connectWifi to each (with long timeout)
 - serial_then_ble: open one serial, then scan/connect BLE for other
 - ble_ble: scan for two BLE devices and connect to both by address

Usage: run from repo root
    python3 yolo/tests/connect_both_wifi.py --ssid YOUR_SSID --password YOUR_PASS

This script prints JSON results from each device and a brief verdict per-method.
"""

import argparse
import sys
import os
import time
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from voxel_sdk.serial import SerialVoxelTransport
from voxel_sdk.ble import BleVoxelTransport
from voxel_sdk.device_controller import DeviceController

# Serial ports on this machine - adjust if needed
SERIAL_PORTS = ['/dev/cu.usbmodem101', '/dev/cu.usbmodem2101']
BLE_NAME = 'voxel'

CONNECT_TIMEOUT = 40  # seconds for connectWifi over serial/BLE


def do_serial_serial(ssid, password):
    print('\n=== METHOD: serial + serial ===')
    results = {}
    transports = []
    controllers = []
    try:
        for i, port in enumerate(SERIAL_PORTS[:2]):
            print(f'Opening serial port {port} (device #{i+1})...')
            t = SerialVoxelTransport(port, baudrate=115200, timeout=CONNECT_TIMEOUT)
            t.connect()
            c = DeviceController(t)
            transports.append(t)
            controllers.append(c)

        # Issue connectWifi on both
        for i, c in enumerate(controllers):
            cmd = f'connectWifi:{ssid}|{password}'
            print(f'Sending connectWifi to serial device #{i+1}...')
            res = c.execute_device_command(cmd)
            results[f'serial_{i+1}'] = res
            print('  ->', res)

    except Exception as e:
        print('Error during serial_serial:', e)
    finally:
        for c in controllers:
            try:
                c.disconnect()
            except Exception:
                pass
        for t in transports:
            try:
                t.disconnect()
            except Exception:
                pass
    return results


def do_serial_then_ble(ssid, password):
    print('\n=== METHOD: serial first, then BLE ===')
    results = {}
    serial_ctrl = None
    ble_ctrl = None
    try:
        # Serial first (use first serial port)
        port = SERIAL_PORTS[0]
        print('Opening serial port', port)
        s = SerialVoxelTransport(port, baudrate=115200, timeout=CONNECT_TIMEOUT)
        s.connect()
        serial_ctrl = DeviceController(s)

        print('Issuing connectWifi on serial device...')
        res_s = serial_ctrl.execute_device_command(f'connectWifi:{ssid}|{password}')
        results['serial'] = res_s
        print('  ->', res_s)

        # Now BLE for the other device (scan)
        print('Scanning and connecting BLE (second device)...')
        b = BleVoxelTransport(device_name=BLE_NAME)
        b.connect("")
        ble_ctrl = DeviceController(b)
        print('Issuing connectWifi on BLE device...')
        res_b = ble_ctrl.execute_device_command(f'connectWifi:{ssid}|{password}')
        results['ble'] = res_b
        print('  ->', res_b)

    except Exception as e:
        print('Error during serial_then_ble:', e)
        results['error'] = str(e)
    finally:
        try:
            if serial_ctrl:
                serial_ctrl.disconnect()
        except Exception:
            pass
        try:
            if ble_ctrl:
                ble_ctrl.disconnect()
        except Exception:
            pass
    return results


def do_ble_ble(ssid, password):
    print('\n=== METHOD: BLE + BLE (scan for addresses) ===')
    results = {}
    controllers = []
    transports = []
    try:
        # First, perform a scan using BleVoxelTransport's underlying scanner
        # We'll create a temporary transport and call connect('') which will scan and pick one address.
        # But to find two addresses we call bleak directly via BleakScanner (import inside to avoid hard dependency failing earlier).
        try:
            from bleak import BleakScanner
        except Exception as e:
            print('bleak not available:', e)
            results['error'] = 'bleak_missing'
            return results

        print('Scanning for BLE devices... (10s)')
        devices = BleakScanner.discover(timeout=6.0)
        # In some bleak versions discover is coroutine; try to handle both
        if hasattr(devices, '__await__'):
            import asyncio
            devices = asyncio.get_event_loop().run_until_complete(devices)

        matches = [d for d in devices if (d.name or '').lower().startswith(BLE_NAME.lower())]
        print(f'Found {len(matches)} matches')
        for d in matches[:2]:
            print(' -', d.address, d.name)

        if len(matches) < 2:
            print('Less than 2 BLE devices found; aborting BLE+BLE test')
            results['found'] = len(matches)
            return results

        # Connect to the two addresses
        for i, dev in enumerate(matches[:2]):
            addr = dev.address
            print(f'Connecting BLE device #{i+1} at address {addr}...')
            b = BleVoxelTransport(device_name=BLE_NAME)
            b.connect(addr)
            transports.append(b)
            controllers.append(DeviceController(b))

        # Issue connectWifi on both
        for i, c in enumerate(controllers):
            print(f'Sending connectWifi to BLE device #{i+1}...')
            res = c.execute_device_command(f'connectWifi:{ssid}|{password}')
            results[f'ble_{i+1}'] = res
            print('  ->', res)

    except Exception as e:
        print('Error during ble_ble:', e)
        results['error'] = str(e)
    finally:
        for c in controllers:
            try:
                c.disconnect()
            except Exception:
                pass
        for t in transports:
            try:
                t.disconnect()
            except Exception:
                pass
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ssid', required=True)
    parser.add_argument('--password', required=True)
    args = parser.parse_args()

    ssid = args.ssid
    passwd = args.password

    # Run methods in order and print JSON summary
    summary = {'serial_serial': None, 'serial_then_ble': None, 'ble_ble': None}

    summary['serial_serial'] = do_serial_serial(ssid, passwd)
    time.sleep(1)
    summary['serial_then_ble'] = do_serial_then_ble(ssid, passwd)
    time.sleep(1)
    summary['ble_ble'] = do_ble_ble(ssid, passwd)

    print('\n=== SUMMARY ===')
    print(json.dumps(summary, indent=2, sort_keys=True))
    print('\nDone')
