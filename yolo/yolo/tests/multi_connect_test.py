"""
Multi-transport connectivity test
- Tries connecting one device over serial and another over BLE in two orders:
  1) Serial first, then BLE
  2) BLE first, then Serial
- Performs 2 attempts for each order. Each attempt allows up to 35 seconds per-device (or until commands succeed).
- Commands executed: get_device_name and list_dir:/

Run from repo root: python3 yolo/tests/multi_connect_test.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from voxel_sdk.serial import SerialVoxelTransport
from voxel_sdk.ble import BleVoxelTransport
from voxel_sdk.device_controller import DeviceController

SERIAL_PORT = '/dev/cu.usbmodem101'
BLE_NAME = 'voxel'
ATTEMPTS = 2
TIMEOUT_PER_DEVICE = 35  # seconds


def run_with_timeout(controller, timeout_seconds):
    """
    Attempt to execute get_device_name and list_dir:/ until success or timeout.
    Returns a dict with results and any errors.
    """
    deadline = time.time() + timeout_seconds
    results = {'get_device_name': None, 'list_dir': None, 'errors': []}

    while time.time() < deadline:
        try:
            if results['get_device_name'] is None:
                res = controller.execute_device_command('get_device_name')
                results['get_device_name'] = res
                print('  -> get_device_name response:', res)
            if results['list_dir'] is None:
                res2 = controller.execute_device_command('list_dir:/')
                results['list_dir'] = res2
                print('  -> list_dir response: (len)' , len(res2.get('files', []) if isinstance(res2, dict) else str(res2)))
            # if we have both, break
            if results['get_device_name'] is not None and results['list_dir'] is not None:
                break
        except Exception as e:
            results['errors'].append(str(e))
            print('  command error:', e)
        time.sleep(0.5)

    return results


def try_serial_then_ble():
    print('\n=== SERIAL FIRST ===')
    for attempt in range(1, ATTEMPTS+1):
        print(f'-- Attempt {attempt} (timeout {TIMEOUT_PER_DEVICE}s per device) --')
        serial_controller = None
        ble_controller = None
        try:
            # Serial
            try:
                s_transport = SerialVoxelTransport(SERIAL_PORT, baudrate=115200, timeout=2)
                s_transport.connect()
                serial_controller = DeviceController(s_transport)
                print('Serial connected to', SERIAL_PORT)
                r = run_with_timeout(serial_controller, TIMEOUT_PER_DEVICE)
                print('Serial results:', r)
            except Exception as e:
                print('Serial connect/command failed:', e)

            # BLE
            try:
                b_transport = BleVoxelTransport(device_name=BLE_NAME)
                b_transport.connect("")
                ble_controller = DeviceController(b_transport)
                print('BLE connected (scanned)')
                r2 = run_with_timeout(ble_controller, TIMEOUT_PER_DEVICE)
                print('BLE results:', r2)
            except Exception as e:
                print('BLE connect/command failed:', e)

        finally:
            try:
                if serial_controller:
                    serial_controller.disconnect()
            except Exception:
                pass
            try:
                if ble_controller:
                    ble_controller.disconnect()
            except Exception:
                pass
        print('-- done attempt --')
        time.sleep(1)


def try_ble_then_serial():
    print('\n=== BLE FIRST ===')
    for attempt in range(1, ATTEMPTS+1):
        print(f'-- Attempt {attempt} (timeout {TIMEOUT_PER_DEVICE}s per device) --')
        serial_controller = None
        ble_controller = None
        try:
            # BLE
            try:
                b_transport = BleVoxelTransport(device_name=BLE_NAME)
                b_transport.connect("")
                ble_controller = DeviceController(b_transport)
                print('BLE connected (scanned)')
                r2 = run_with_timeout(ble_controller, TIMEOUT_PER_DEVICE)
                print('BLE results:', r2)
            except Exception as e:
                print('BLE connect/command failed:', e)

            # Serial
            try:
                s_transport = SerialVoxelTransport(SERIAL_PORT, baudrate=115200, timeout=2)
                s_transport.connect()
                serial_controller = DeviceController(s_transport)
                print('Serial connected to', SERIAL_PORT)
                r = run_with_timeout(serial_controller, TIMEOUT_PER_DEVICE)
                print('Serial results:', r)
            except Exception as e:
                print('Serial connect/command failed:', e)

        finally:
            try:
                if serial_controller:
                    serial_controller.disconnect()
            except Exception:
                pass
            try:
                if ble_controller:
                    ble_controller.disconnect()
            except Exception:
                pass
        print('-- done attempt --')
        time.sleep(1)


if __name__ == '__main__':
    print('Starting multi-transport connectivity test')
    try_serial_then_ble()
    try_ble_then_serial()
    print('Test complete')
