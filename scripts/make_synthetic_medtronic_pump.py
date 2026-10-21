#!/usr/bin/env -S uv run python
"""Write the synthetic MiniMed pump CareLink export, data/input/medtronic_pump_synthetic.csv.

Hand-authored, not scrubbed: every row below is invented, one or two per kind of record a real 720G
export carries (docs/PLAN_pump.md §2.2), so tests/test_medtronic_pump.py can state its expected values
by hand from this list. The file layout (BOM, five metadata lines, a "-------;<device>;<section>" line
before each section, the 54-column header repeated per section, comma decimals) follows the real
exports. Dates are 1961-04-12, the other synthetic fixtures' date.

    uv run scripts/make_synthetic_medtronic_pump.py

Edit the ROWS below, rerun, and update the test's hand-written expectations with it.
"""

from pathlib import Path

from cgm_format.formats.medtronic_pump import MedtronicPumpColumn as C

OUT = Path(__file__).resolve().parent.parent / "data" / "input" / "medtronic_pump_synthetic.csv"
DATE = "1961/04/12"
DEVICE = "MiniMed 720G MMT-1810"

# (time, {column: cell}) in the order the pump logged them; Index follows this order.
PUMP_ROWS: list[tuple[str, dict[C, str]]] = [
    ("00:00:00", {C.BASAL_RATE: "1,000"}),
    # A Percent temp: the log keeps the scheduled 1,000; 50 % applies for an hour.
    ("01:00:00", {C.TEMP_BASAL_AMOUNT: "50", C.TEMP_BASAL_TYPE: "Percent", C.TEMP_BASAL_DURATION: "01:00:00"}),
    ("01:00:00", {C.BASAL_RATE: "1,000"}),
    # A schedule change inside the percent temp: 0,800 at 50 %.
    ("01:30:00", {C.BASAL_RATE: "0,800"}),
    ("02:00:00", {C.BASAL_RATE: "0,800"}),
    # A Rate temp: the log already carries the temp rate.
    ("03:00:00", {C.TEMP_BASAL_AMOUNT: "2,000", C.TEMP_BASAL_TYPE: "Rate", C.TEMP_BASAL_DURATION: "00:30:00"}),
    ("03:00:00", {C.BASAL_RATE: "2,000"}),
    ("03:30:00", {C.BASAL_RATE: "0,800"}),
    # Two rates logged in one second: the later one is in force.
    ("04:00:00", {C.BASAL_RATE: "0,900"}),
    ("04:00:00", {C.BASAL_RATE: "1,100"}),
    # A user suspend logs a 0; a rewind and two primings happen while suspended.
    ("05:00:00", {C.SUSPEND: "USER_SUSPEND"}),
    ("05:00:00", {C.BASAL_RATE: "0,000"}),
    ("05:10:00", {C.REWIND: "Rewind"}),
    ("05:12:00", {C.PRIME_TYPE: "Tubing Fill", C.PRIME_VOLUME_DELIVERED: "10,000", C.ESTIMATED_RESERVOIR_VOLUME: "180,000"}),
    ("05:13:00", {C.PRIME_TYPE: "Cannula Fill", C.PRIME_VOLUME_DELIVERED: "0,300"}),
    ("05:15:00", {C.SUSPEND: "NORMAL_PUMPING"}),
    ("05:15:00", {C.BASAL_RATE: "1,100"}),
    # An alarm suspend logs no 0, and its resume logs no rate.
    ("06:00:00", {C.SUSPEND: "ALARM_SUSPEND", C.ALERT: "INSERT BATTERY ALARM, DELIVERY STOPPED"}),
    ("06:20:00", {C.SUSPEND: "NORMAL_PUMPING"}),
    # Bolus 1, normal: programmed, then delivered 90 s later.
    ("07:00:00", {C.BOLUS_TYPE: "Normal", C.BOLUS_VOLUME_SELECTED: "3,000", C.BOLUS_NUMBER: "1", C.BOLUS_SOURCE: "MANUAL"}),
    ("07:01:30", {C.BOLUS_TYPE: "Normal", C.BOLUS_VOLUME_SELECTED: "3,000", C.BOLUS_VOLUME_DELIVERED: "3,000", C.BOLUS_NUMBER: "1"}),
    # Bolus 2, dual: 2,000 at once and 1,500 over an hour, the square part's delivered row at its end.
    ("08:00:00", {C.BOLUS_TYPE: "Dual (square part)", C.BOLUS_VOLUME_SELECTED: "1,500", C.BOLUS_DURATION: "01:00:00", C.BOLUS_NUMBER: "2"}),
    ("08:00:00", {C.BOLUS_TYPE: "Dual (normal part)", C.BOLUS_VOLUME_SELECTED: "2,000", C.BOLUS_NUMBER: "2"}),
    ("08:00:40", {C.BOLUS_TYPE: "Dual (normal part)", C.BOLUS_VOLUME_SELECTED: "2,000", C.BOLUS_VOLUME_DELIVERED: "2,000", C.BOLUS_NUMBER: "2"}),
    ("09:00:00", {C.BOLUS_TYPE: "Dual (square part)", C.BOLUS_VOLUME_SELECTED: "1,500", C.BOLUS_VOLUME_DELIVERED: "1,500", C.BOLUS_DURATION: "01:00:00", C.BOLUS_NUMBER: "2"}),
    # Bolus 3, square, cancelled after 20 of 30 minutes and reporting the 20.
    ("10:00:00", {C.BOLUS_TYPE: "Square", C.BOLUS_VOLUME_SELECTED: "2,000", C.BOLUS_DURATION: "00:30:00", C.BOLUS_NUMBER: "3"}),
    ("10:20:00", {C.BOLUS_TYPE: "Square", C.BOLUS_VOLUME_SELECTED: "2,000", C.BOLUS_VOLUME_DELIVERED: "1,000", C.BOLUS_DURATION: "00:20:00", C.BOLUS_NUMBER: "3", C.BOLUS_CANCELLATION_REASON: "User Request"}),
    # Bolus 4, square, short of its volume but reporting the programmed 30 minutes: the warned case.
    ("11:00:00", {C.BOLUS_TYPE: "Square", C.BOLUS_VOLUME_SELECTED: "2,000", C.BOLUS_DURATION: "00:30:00", C.BOLUS_NUMBER: "4"}),
    ("11:30:00", {C.BOLUS_TYPE: "Square", C.BOLUS_VOLUME_SELECTED: "2,000", C.BOLUS_VOLUME_DELIVERED: "1,500", C.BOLUS_DURATION: "00:30:00", C.BOLUS_NUMBER: "4"}),
    # Bolus 5, cancelled before any insulin left the pump.
    ("12:00:00", {C.BOLUS_TYPE: "Normal", C.BOLUS_VOLUME_SELECTED: "1,000", C.BOLUS_NUMBER: "5"}),
    ("12:00:30", {C.BOLUS_TYPE: "Normal", C.BOLUS_VOLUME_SELECTED: "1,000", C.BOLUS_VOLUME_DELIVERED: "0,000", C.BOLUS_NUMBER: "5", C.BOLUS_CANCELLATION_REASON: "User Request"}),
    # A bolus type this parser does not know.
    ("12:30:00", {C.BOLUS_TYPE: "Mystery", C.BOLUS_VOLUME_SELECTED: "1,000", C.BOLUS_VOLUME_DELIVERED: "1,000", C.BOLUS_NUMBER: "6"}),
    # A wizard meal: carbs, a fingerstick, and bolus 7.
    ("13:00:00", {C.BG_SOURCE: "ENTERED_IN_BOLUS_WIZRD", C.BG_READING: "120"}),
    ("13:00:00", {C.BWZ_CARB_INPUT: "45", C.BWZ_BG_INPUT: "120", C.BWZ_ESTIMATE: "4,500", C.BWZ_STATUS: "Delivered"}),
    ("13:00:00", {C.BOLUS_TYPE: "Normal", C.BOLUS_VOLUME_SELECTED: "4,500", C.BOLUS_NUMBER: "7", C.BOLUS_SOURCE: "BOLUS_WIZARD"}),
    ("13:00:20", {C.BOLUS_TYPE: "Normal", C.BOLUS_VOLUME_SELECTED: "4,500", C.BOLUS_VOLUME_DELIVERED: "4,500", C.BOLUS_NUMBER: "7"}),
    # A fingerstick entered outside the wizard, and a wizard run with no carbs.
    ("14:00:00", {C.BG_SOURCE: "ENTERED_IN_BG_ENTRY", C.BG_READING: "95"}),
    ("15:00:00", {C.BWZ_CARB_INPUT: "0", C.BWZ_BG_INPUT: "0", C.BWZ_ESTIMATE: "0,000", C.BWZ_STATUS: "Not Delivered"}),
    ("16:00:00", {C.BASAL_RATE: "1,000"}),
]

SENSOR_ROWS: list[tuple[str, dict[C, str]]] = [
    ("00:00:00", {C.EVENT_MARKER: "Start of the day"}),
    ("00:05:00", {C.SENSOR_GLUCOSE: "110", C.ISIG_VALUE: "20,10"}),
    ("23:59:59", {C.EVENT_MARKER: "End of the day"}),
]

COLUMNS = list(C)
HEADER = ";".join(c.value for c in COLUMNS)


def section(name: str, rows: list[tuple[str, dict[C, str]]], first_index: int) -> list[str]:
    lines = [f"-------;{DEVICE};{name}", HEADER]
    for k, (time, cells) in enumerate(rows):
        base = {C.INDEX: f"{first_index + k},00000", C.DATE: DATE, C.TIME: time}
        lines.append(";".join({**base, **cells}.get(c, "") for c in COLUMNS))
    return lines


def main() -> None:
    lines = [
        "Last Name;First Name;Patient ID;Start Date;End Date",
        '"Gagarin";"Yuri";"";"12.04.1961";"12.04.1961"',
        "Patient DOB;;;;",
        "",
        "",
        *section("Pump", PUMP_ROWS, 0),
        *section("Sensor", SENSOR_ROWS, len(PUMP_ROWS)),
    ]
    OUT.write_text("﻿" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(PUMP_ROWS)} pump rows, {len(SENSOR_ROWS)} sensor rows)")


if __name__ == "__main__":
    main()
