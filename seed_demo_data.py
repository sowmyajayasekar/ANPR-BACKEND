"""
Seed script to populate clearly labelled DEMO data for college presentation.
Includes toll gates, vehicle owner profiles, purchase history, and sample toll detections.
Safe to run multiple times without duplicating or overwriting genuine records.

ALL DATA IN THIS SCRIPT IS DEMO/SAMPLE DATA FOR ACADEMIC DEMONSTRATION ONLY.
It does not represent real vehicle owners or actual government records.
"""

import sqlite3
import os

DATABASE = "anpr_history.db"


def seed_demo_data():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # ---- 1. Toll Gates -------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS toll_gates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            highway TEXT,
            city TEXT,
            state TEXT,
            latitude REAL,
            longitude REAL,
            address TEXT,
            is_demo INTEGER DEFAULT 1
        )
    """)

    demo_toll_gates = [
        (1, "Coimbatore North Toll Plaza (DEMO)", "NH 544", "Coimbatore",
         "Tamil Nadu", 11.0526, 76.9935,
         "Salem-Cochin Highway, Coimbatore, TN", 1),
        (2, "Tiruppur Ring Road Toll Plaza (DEMO)", "SH 169", "Tiruppur",
         "Tamil Nadu", 11.1085, 77.3411,
         "Perumanallur Bypass, Tiruppur, TN", 1),
        (3, "Hosur Border Toll Plaza (DEMO)", "NH 44", "Hosur",
         "Tamil Nadu", 12.7409, 77.8253,
         "Bengaluru-Chennai Highway, Hosur, TN", 1),
        (4, "Walayar Border Toll Plaza (DEMO)", "NH 544", "Palakkad",
         "Kerala", 10.8415, 76.8488,
         "Kerala-TN Border Post, Walayar, KL", 1),
        (5, "Electronic City Toll Plaza (DEMO)", "NH 44", "Bengaluru",
         "Karnataka", 12.8452, 77.6602,
         "Hosur Elevated Expressway, Bengaluru, KA", 1),
        (6, "Khalapur Toll Plaza (DEMO)", "Expressway", "Pune",
         "Maharashtra", 18.7844, 73.2842,
         "Mumbai-Pune Expressway, Khalapur, MH", 1),
    ]

    for tg in demo_toll_gates:
        cursor.execute("""
            INSERT INTO toll_gates
                (id, name, highway, city, state, latitude, longitude, address, is_demo)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name,
                highway=excluded.highway,
                city=excluded.city,
                state=excluded.state,
                latitude=excluded.latitude,
                longitude=excluded.longitude,
                address=excluded.address
        """, tg)

    # ---- 2. Vehicles (Extended schema) ---------------------------------------
    existing_cols = {row[1] for row in
                     cursor.execute("PRAGMA table_info(vehicles)").fetchall()}
    new_vehicle_cols = {
        "state_code": "TEXT",
        "state_name": "TEXT",
        "rto_code": "TEXT",
        "rto_name": "TEXT",
        "manufacturer_model": "TEXT",
        "fuel_type": "TEXT",
        "vehicle_color": "TEXT",
        "registration_year": "INTEGER",
        "registration_date": "TEXT",
        "purchase_date": "TEXT",
        "purchase_condition": "TEXT",
        "previous_owners_count": "INTEGER DEFAULT 0",
        "previous_ownership_history": "TEXT",
        "insurance_expiry": "TEXT",
        "fitness_expiry": "TEXT",
        "pollution_expiry": "TEXT",
        "is_demo": "INTEGER DEFAULT 0",
    }
    for col, col_type in new_vehicle_cols.items():
        if col not in existing_cols:
            cursor.execute(f"ALTER TABLE vehicles ADD COLUMN {col} {col_type}")

    demo_vehicles = [
        {
            "plate_number": "TN39AB1234",
            "owner_name": "Karthik Subramanian (DEMO)",
            "registration_status": "Active",
            "state_code": "TN", "state_name": "Tamil Nadu",
            "rto_code": "TN 39", "rto_name": "Tiruppur North",
            "manufacturer_model": "Hyundai Creta SX (O)",
            "fuel_type": "Diesel", "vehicle_color": "Polar White",
            "registration_year": 2021, "registration_date": "2021-03-15",
            "purchase_date": "2021-03-10", "purchase_condition": "New",
            "previous_owners_count": 0,
            "previous_ownership_history": "First owner directly registered from Chandra Hyundai, Tiruppur.",
            "insurance_expiry": "2027-03-14",
            "fitness_expiry": "2036-03-14",
            "pollution_expiry": "2027-01-20",
            "notes": "DEMO RECORD — Tamil Nadu / Tiruppur vehicle",
            "is_demo": 1,
        },
        {
            "plate_number": "TN42CD5678",
            "owner_name": "Lakshmi Narayanan (DEMO)",
            "registration_status": "Active",
            "state_code": "TN", "state_name": "Tamil Nadu",
            "rto_code": "TN 42", "rto_name": "Tiruppur South",
            "manufacturer_model": "Maruti Suzuki Swift VXi",
            "fuel_type": "Petrol", "vehicle_color": "Magma Grey",
            "registration_year": 2019, "registration_date": "2019-07-22",
            "purchase_date": "2022-09-18", "purchase_condition": "Second-hand",
            "previous_owners_count": 1,
            "previous_ownership_history": "Transferred from R. Suresh (Coimbatore) on 2022-09-18.",
            "insurance_expiry": "2026-11-30",
            "fitness_expiry": "2034-07-21",
            "pollution_expiry": "2026-12-15",
            "notes": "DEMO RECORD — Second-hand purchase example",
            "is_demo": 1,
        },
        {
            "plate_number": "TN38MN9012",
            "owner_name": "Dr. Anand Rajagopal (DEMO)",
            "registration_status": "Active",
            "state_code": "TN", "state_name": "Tamil Nadu",
            "rto_code": "TN 38", "rto_name": "Coimbatore North",
            "manufacturer_model": "Tata Nexon EV Max",
            "fuel_type": "Electric", "vehicle_color": "Pristine White",
            "registration_year": 2023, "registration_date": "2023-01-10",
            "purchase_date": "2023-01-05", "purchase_condition": "New",
            "previous_owners_count": 0,
            "previous_ownership_history": "First owner registered via SRT Motors, Coimbatore.",
            "insurance_expiry": "2028-01-09",
            "fitness_expiry": "2038-01-09",
            "pollution_expiry": "Not Applicable (EV)",
            "notes": "DEMO RECORD — Coimbatore Electric Vehicle",
            "is_demo": 1,
        },
        {
            "plate_number": "KL07CD3456",
            "owner_name": "Mathew Varghese (DEMO)",
            "registration_status": "Active",
            "state_code": "KL", "state_name": "Kerala",
            "rto_code": "KL 07", "rto_name": "Ernakulam / Kochi",
            "manufacturer_model": "Toyota Innova Crysta 2.4Z",
            "fuel_type": "Diesel", "vehicle_color": "Silver Metallic",
            "registration_year": 2020, "registration_date": "2020-05-18",
            "purchase_date": "2020-05-12", "purchase_condition": "New",
            "previous_owners_count": 0,
            "previous_ownership_history": "First owner registered at Nippon Toyota, Kalamassery.",
            "insurance_expiry": "2027-05-17",
            "fitness_expiry": "2035-05-17",
            "pollution_expiry": "2026-11-10",
            "notes": "DEMO RECORD — Kerala interstate travel example",
            "is_demo": 1,
        },
        {
            "plate_number": "KA03MN7890",
            "owner_name": "Pooja Venkatesh (DEMO)",
            "registration_status": "Active",
            "state_code": "KA", "state_name": "Karnataka",
            "rto_code": "KA 03", "rto_name": "Indiranagar / Bengaluru East",
            "manufacturer_model": "Honda City 1.5 i-VTEC",
            "fuel_type": "Petrol", "vehicle_color": "Radiant Red",
            "registration_year": 2022, "registration_date": "2022-08-04",
            "purchase_date": "2022-07-28", "purchase_condition": "New",
            "previous_owners_count": 0,
            "previous_ownership_history": "First owner registered via Whitefield Honda, Bengaluru.",
            "insurance_expiry": "2027-08-03",
            "fitness_expiry": "2037-08-03",
            "pollution_expiry": "2027-02-14",
            "notes": "DEMO RECORD — Karnataka RTO registration",
            "is_demo": 1,
        },
        {
            "plate_number": "MH47TC0964",
            "owner_name": "Rohan Kulkarni (DEMO)",
            "registration_status": "Active",
            "state_code": "MH", "state_name": "Maharashtra",
            "rto_code": "MH 47", "rto_name": "Mumbai North / Borivali",
            "manufacturer_model": "Mahindra XUV700 AX7",
            "fuel_type": "Diesel", "vehicle_color": "Midnight Black",
            "registration_year": 2022, "registration_date": "2022-11-20",
            "purchase_date": "2022-11-15", "purchase_condition": "New",
            "previous_owners_count": 0,
            "previous_ownership_history": "First owner registered via NBS International, Mumbai.",
            "insurance_expiry": "2027-11-19",
            "fitness_expiry": "2037-11-19",
            "pollution_expiry": "2026-10-30",
            "notes": "DEMO RECORD — Maharashtra / Mumbai North vehicle",
            "is_demo": 1,
        },
        {
            "plate_number": "MH12AB1234",
            "owner_name": "Priya Deshmukh (DEMO)",
            "registration_status": "Active",
            "state_code": "MH", "state_name": "Maharashtra",
            "rto_code": "MH 12", "rto_name": "Pune Central",
            "manufacturer_model": "Bajaj Pulsar 150",
            "fuel_type": "Petrol", "vehicle_color": "Black / Red",
            "registration_year": 2020, "registration_date": "2020-03-08",
            "purchase_date": "2020-03-02", "purchase_condition": "New",
            "previous_owners_count": 0,
            "previous_ownership_history": "First owner from Bajaj dealer, Pune.",
            "insurance_expiry": "2026-03-07",
            "fitness_expiry": "2030-03-07",
            "pollution_expiry": "2026-09-12",
            "notes": "DEMO RECORD — Pune Maharashtra example",
            "is_demo": 1,
        },
    ]

    for v in demo_vehicles:
        cursor.execute("""
            INSERT INTO vehicles (
                plate_number, owner_name, registration_status, notes,
                state_code, state_name, rto_code, rto_name,
                manufacturer_model, fuel_type, vehicle_color,
                registration_year, registration_date, purchase_date,
                purchase_condition, previous_owners_count,
                previous_ownership_history, insurance_expiry,
                fitness_expiry, pollution_expiry, is_demo
            )
            VALUES (
                :plate_number, :owner_name, :registration_status, :notes,
                :state_code, :state_name, :rto_code, :rto_name,
                :manufacturer_model, :fuel_type, :vehicle_color,
                :registration_year, :registration_date, :purchase_date,
                :purchase_condition, :previous_owners_count,
                :previous_ownership_history, :insurance_expiry,
                :fitness_expiry, :pollution_expiry, :is_demo
            )
            ON CONFLICT(plate_number) DO UPDATE SET
                owner_name=excluded.owner_name,
                registration_status=excluded.registration_status,
                state_code=excluded.state_code,
                state_name=excluded.state_name,
                rto_code=excluded.rto_code,
                rto_name=excluded.rto_name,
                manufacturer_model=excluded.manufacturer_model,
                fuel_type=excluded.fuel_type,
                vehicle_color=excluded.vehicle_color,
                registration_year=excluded.registration_year,
                registration_date=excluded.registration_date,
                purchase_date=excluded.purchase_date,
                purchase_condition=excluded.purchase_condition,
                previous_owners_count=excluded.previous_owners_count,
                previous_ownership_history=excluded.previous_ownership_history,
                insurance_expiry=excluded.insurance_expiry,
                fitness_expiry=excluded.fitness_expiry,
                pollution_expiry=excluded.pollution_expiry,
                is_demo=excluded.is_demo
        """, v)

    # ---- 3. Detections — Extend Schema ---------------------------------------
    existing_det_cols = {row[1] for row in
                         cursor.execute("PRAGMA table_info(detections)").fetchall()}
    new_det_cols = {
        "state_code": "TEXT",
        "state_name": "TEXT",
        "rto_code": "TEXT",
        "rto_name": "TEXT",
        "toll_gate_id": "INTEGER",
        "toll_gate_name": "TEXT",
        "validation_status": "TEXT",
        "display_plate": "TEXT",
        "is_demo": "INTEGER DEFAULT 0",
    }
    for col, col_type in new_det_cols.items():
        if col not in existing_det_cols:
            cursor.execute(f"ALTER TABLE detections ADD COLUMN {col} {col_type}")

    # ---- 4. Seed Demo Detections (only if none exist yet) --------------------
    demo_count = cursor.execute(
        "SELECT COUNT(*) FROM detections WHERE is_demo = 1"
    ).fetchone()[0]

    if demo_count == 0:
        demo_detections = [
            # TN39AB1234 — multiple detections at different toll gates
            {
                "image_name": "demo_tn39_cam1.jpg",
                "plate_number": "TN39AB1234",
                "display_plate": "TN 39 AB 1234",
                "ocr_confidence": 98.7,
                "best_method": "CLAHE",
                "status": "Recognized",
                "location": "Coimbatore North Toll Plaza (DEMO)",
                "detection_time": "2026-10-01 08:32:14",
                "plate_quality": "Good",
                "alert": "None",
                "watchlist_match": 0,
                "watchlist_reason": None,
                "previous_detections": 0,
                "latitude": 11.0526, "longitude": 76.9935,
                "original_image": None, "plate_image": None,
                "state_code": "TN", "state_name": "Tamil Nadu",
                "rto_code": "TN 39", "rto_name": "Tiruppur North",
                "toll_gate_id": 1,
                "toll_gate_name": "Coimbatore North Toll Plaza (DEMO)",
                "validation_status": "Valid Indian Format",
                "is_demo": 1,
            },
            {
                "image_name": "demo_tn39_cam2.jpg",
                "plate_number": "TN39AB1234",
                "display_plate": "TN 39 AB 1234",
                "ocr_confidence": 96.4,
                "best_method": "Sharpened",
                "status": "Recognized",
                "location": "Tiruppur Ring Road Toll Plaza (DEMO)",
                "detection_time": "2026-10-04 14:15:22",
                "plate_quality": "Good",
                "alert": "WATCHLIST MATCH",
                "watchlist_match": 1,
                "watchlist_reason": "Speed Violations & Toll Evasion Notice (DEMO)",
                "previous_detections": 1,
                "latitude": 11.1085, "longitude": 77.3411,
                "original_image": None, "plate_image": None,
                "state_code": "TN", "state_name": "Tamil Nadu",
                "rto_code": "TN 39", "rto_name": "Tiruppur North",
                "toll_gate_id": 2,
                "toll_gate_name": "Tiruppur Ring Road Toll Plaza (DEMO)",
                "validation_status": "Valid Indian Format",
                "is_demo": 1,
            },
            {
                "image_name": "demo_tn39_cam3.jpg",
                "plate_number": "TN39AB1234",
                "display_plate": "TN 39 AB 1234",
                "ocr_confidence": 97.1,
                "best_method": "Denoised",
                "status": "Recognized",
                "location": "Hosur Border Toll Plaza (DEMO)",
                "detection_time": "2026-10-08 19:50:30",
                "plate_quality": "Good",
                "alert": "WATCHLIST MATCH",
                "watchlist_match": 1,
                "watchlist_reason": "Speed Violations & Toll Evasion Notice (DEMO)",
                "previous_detections": 2,
                "latitude": 12.7409, "longitude": 77.8253,
                "original_image": None, "plate_image": None,
                "state_code": "TN", "state_name": "Tamil Nadu",
                "rto_code": "TN 39", "rto_name": "Tiruppur North",
                "toll_gate_id": 3,
                "toll_gate_name": "Hosur Border Toll Plaza (DEMO)",
                "validation_status": "Valid Indian Format",
                "is_demo": 1,
            },
            # TN42CD5678
            {
                "image_name": "demo_tn42_cam1.jpg",
                "plate_number": "TN42CD5678",
                "display_plate": "TN 42 CD 5678",
                "ocr_confidence": 94.2,
                "best_method": "Otsu",
                "status": "Recognized",
                "location": "Tiruppur Ring Road Toll Plaza (DEMO)",
                "detection_time": "2026-10-05 11:20:05",
                "plate_quality": "Good",
                "alert": "None",
                "watchlist_match": 0,
                "watchlist_reason": None,
                "previous_detections": 0,
                "latitude": 11.1085, "longitude": 77.3411,
                "original_image": None, "plate_image": None,
                "state_code": "TN", "state_name": "Tamil Nadu",
                "rto_code": "TN 42", "rto_name": "Tiruppur South",
                "toll_gate_id": 2,
                "toll_gate_name": "Tiruppur Ring Road Toll Plaza (DEMO)",
                "validation_status": "Valid Indian Format",
                "is_demo": 1,
            },
            {
                "image_name": "demo_tn42_cam2.jpg",
                "plate_number": "TN42CD5678",
                "display_plate": "TN 42 CD 5678",
                "ocr_confidence": 91.3,
                "best_method": "Adaptive",
                "status": "Recognized",
                "location": "Coimbatore North Toll Plaza (DEMO)",
                "detection_time": "2026-10-10 07:05:18",
                "plate_quality": "Good",
                "alert": "None",
                "watchlist_match": 0,
                "watchlist_reason": None,
                "previous_detections": 1,
                "latitude": 11.0526, "longitude": 76.9935,
                "original_image": None, "plate_image": None,
                "state_code": "TN", "state_name": "Tamil Nadu",
                "rto_code": "TN 42", "rto_name": "Tiruppur South",
                "toll_gate_id": 1,
                "toll_gate_name": "Coimbatore North Toll Plaza (DEMO)",
                "validation_status": "Valid Indian Format",
                "is_demo": 1,
            },
            # KL07CD3456
            {
                "image_name": "demo_kl07_cam1.jpg",
                "plate_number": "KL07CD3456",
                "display_plate": "KL 07 CD 3456",
                "ocr_confidence": 92.8,
                "best_method": "Denoised",
                "status": "Recognized",
                "location": "Walayar Border Toll Plaza (DEMO)",
                "detection_time": "2026-10-06 09:45:10",
                "plate_quality": "Good",
                "alert": "None",
                "watchlist_match": 0,
                "watchlist_reason": None,
                "previous_detections": 0,
                "latitude": 10.8415, "longitude": 76.8488,
                "original_image": None, "plate_image": None,
                "state_code": "KL", "state_name": "Kerala",
                "rto_code": "KL 07", "rto_name": "Ernakulam / Kochi",
                "toll_gate_id": 4,
                "toll_gate_name": "Walayar Border Toll Plaza (DEMO)",
                "validation_status": "Valid Indian Format",
                "is_demo": 1,
            },
            # KA03MN7890
            {
                "image_name": "demo_ka03_cam1.jpg",
                "plate_number": "KA03MN7890",
                "display_plate": "KA 03 MN 7890",
                "ocr_confidence": 88.5,
                "best_method": "CLAHE",
                "status": "Recognized",
                "location": "Electronic City Toll Plaza (DEMO)",
                "detection_time": "2026-10-06 17:30:45",
                "plate_quality": "Moderate",
                "alert": "None",
                "watchlist_match": 0,
                "watchlist_reason": None,
                "previous_detections": 0,
                "latitude": 12.8452, "longitude": 77.6602,
                "original_image": None, "plate_image": None,
                "state_code": "KA", "state_name": "Karnataka",
                "rto_code": "KA 03", "rto_name": "Indiranagar / Bengaluru East",
                "toll_gate_id": 5,
                "toll_gate_name": "Electronic City Toll Plaza (DEMO)",
                "validation_status": "Valid Indian Format",
                "is_demo": 1,
            },
            # MH47TC0964
            {
                "image_name": "demo_mh47_cam1.jpg",
                "plate_number": "MH47TC0964",
                "display_plate": "MH 47 TC 0964",
                "ocr_confidence": 97.8,
                "best_method": "CLAHE_High",
                "status": "Recognized",
                "location": "Khalapur Toll Plaza (DEMO)",
                "detection_time": "2026-10-07 12:10:30",
                "plate_quality": "Good",
                "alert": "None",
                "watchlist_match": 0,
                "watchlist_reason": None,
                "previous_detections": 0,
                "latitude": 18.7844, "longitude": 73.2842,
                "original_image": None, "plate_image": None,
                "state_code": "MH", "state_name": "Maharashtra",
                "rto_code": "MH 47", "rto_name": "Mumbai North / Borivali",
                "toll_gate_id": 6,
                "toll_gate_name": "Khalapur Toll Plaza (DEMO)",
                "validation_status": "Valid Indian Format",
                "is_demo": 1,
            },
            # Low confidence example
            {
                "image_name": "demo_low_conf_cam1.jpg",
                "plate_number": "TN38MN9012",
                "display_plate": "TN 38 MN 9012",
                "ocr_confidence": 48.2,
                "best_method": "Adaptive",
                "status": "Low Confidence",
                "location": "Coimbatore North Toll Plaza (DEMO)",
                "detection_time": "2026-10-07 07:12:00",
                "plate_quality": "Poor",
                "alert": "Manual verification recommended",
                "watchlist_match": 0,
                "watchlist_reason": None,
                "previous_detections": 0,
                "latitude": 11.0526, "longitude": 76.9935,
                "original_image": None, "plate_image": None,
                "state_code": "TN", "state_name": "Tamil Nadu",
                "rto_code": "TN 38", "rto_name": "Coimbatore North",
                "toll_gate_id": 1,
                "toll_gate_name": "Coimbatore North Toll Plaza (DEMO)",
                "validation_status": "Manual Verification Recommended",
                "is_demo": 1,
            },
            # Plate not detected example
            {
                "image_name": "demo_not_detected.jpg",
                "plate_number": "Not Detected",
                "display_plate": "Not Detected",
                "ocr_confidence": 0.0,
                "best_method": "None",
                "status": "Detection Failed",
                "location": "Tiruppur Ring Road Toll Plaza (DEMO)",
                "detection_time": "2026-10-08 16:45:10",
                "plate_quality": "Very Poor",
                "alert": "No number plate detected",
                "watchlist_match": 0,
                "watchlist_reason": None,
                "previous_detections": 0,
                "latitude": 11.1085, "longitude": 77.3411,
                "original_image": None, "plate_image": None,
                "state_code": "Unverified", "state_name": "Unverified",
                "rto_code": "Unverified", "rto_name": "Unverified",
                "toll_gate_id": 2,
                "toll_gate_name": "Tiruppur Ring Road Toll Plaza (DEMO)",
                "validation_status": "Plate Not Detected",
                "is_demo": 1,
            },
        ]

        for det in demo_detections:
            cursor.execute("""
                INSERT INTO detections (
                    image_name, plate_number, display_plate,
                    ocr_confidence, best_method, status,
                    location, detection_time, plate_quality, alert,
                    watchlist_match, watchlist_reason, previous_detections,
                    latitude, longitude, original_image, plate_image,
                    state_code, state_name, rto_code, rto_name,
                    toll_gate_id, toll_gate_name, validation_status, is_demo
                )
                VALUES (
                    :image_name, :plate_number, :display_plate,
                    :ocr_confidence, :best_method, :status,
                    :location, :detection_time, :plate_quality, :alert,
                    :watchlist_match, :watchlist_reason, :previous_detections,
                    :latitude, :longitude, :original_image, :plate_image,
                    :state_code, :state_name, :rto_code, :rto_name,
                    :toll_gate_id, :toll_gate_name, :validation_status, :is_demo
                )
            """, det)

    # ---- 5. Watchlist Demo Entry --------------------------------------------
    cursor.execute("""
        INSERT INTO watchlist (plate_number, reason, added_time)
        VALUES ('TN39AB1234',
                'Speed Violations & Toll Evasion Notice (DEMO)',
                '2026-10-01 10:00:00')
        ON CONFLICT(plate_number) DO NOTHING
    """)

    conn.commit()
    conn.close()
    print("=" * 50)
    print("  Database seeded with DEMO data successfully!")
    print("  ALL SEEDED RECORDS ARE CLEARLY MARKED is_demo=1")
    print("  They do NOT represent real vehicle owners or")
    print("  actual government data.")
    print("=" * 50)


if __name__ == "__main__":
    seed_demo_data()
