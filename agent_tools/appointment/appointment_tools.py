import os
import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime, timedelta
from langchain_core.tools import tool

from utils import get_cwd

print ("\n",get_cwd(), "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\n")

# Load appointment data
appointments_df = pd.read_csv(get_cwd() + os.sep + 'data' + os.sep + 'appointments.csv')
appointments_df['datetime'] = pd.to_datetime(appointments_df['datetime'])


def generate_available_slots(specialist_id: str, start_date: str, end_date: str) -> List[dict]:
    """Generate available time slots for a specialist based on the new logic."""
    try:
        # Convert string dates to datetime objects
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        available_slots = []
        current_date = start_dt
        
        # Generate slots for each day in the range
        while current_date <= end_dt:
            # Skip weekends (Saturday=5, Sunday=6)
            if current_date.weekday() < 5:
                # Generate time slots for the day (11:00 to 18:00)
                for hour in [11, 12, 13, 15, 16, 17]:  # Skip 14:00-15:00 (lunch)
                    for minute in [0, 30]:  # 30-minute intervals
                        slot_time = current_date.replace(hour=hour, minute=minute, second=0)
                        
                        # Check if this slot is already booked in appointments.csv
                        is_booked = not appointments_df[
                            (appointments_df['specialist_id'] == specialist_id) &
                            (appointments_df['datetime'] == slot_time) &
                            (appointments_df['status'] == 'booked')
                        ].empty
                        
                        if not is_booked:
                            available_slots.append({
                                'slot_id': f"SLOT-{specialist_id}-{slot_time.strftime('%Y%m%d%H%M')}",
                                'specialist_id': specialist_id,
                                'datetime': slot_time.strftime('%Y-%m-%d %H:%M:%S'),
                                'duration_minutes': 30,
                                'status': 'available'
                            })
            
            current_date += timedelta(days=1)
        
        return available_slots
        
    except Exception as e:
        print(f"Error generating slots: {e}")
        return []

@tool
def check_appointment_availability(specialist_id: Optional[str] = None, 
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None) -> List[dict]:
    """Check available appointments for specific specialists. 
    Returns available slots between 11:00-18:00 (excluding 14:00-15:00 lunch break) on weekdays.
    Date format: YYYY-MM-DD. If no dates provided, defaults to next 7 days."""
    print ("\ncheck_appointment_availability called with: ", specialist_id, start_date, end_date)

    try:
        # Set default date range (next 7 days)
        today = datetime.now().date()
        if not start_date:
            start_date = today.strftime('%Y-%m-%d')
        if not end_date:
            end_date = (today + timedelta(days=7)).strftime('%Y-%m-%d')
        
        # If no specialist_id specified, get all doctors from the data
        if not specialist_id:
            # You might want to load doctors here or get from another source
            specialist_ids = ['ps-301', 'ps-302']  # Example doctor IDs
        else:
            specialist_ids = [specialist_id]
        
        all_available_slots = []
        
        for doc_id in specialist_ids:
            slots = generate_available_slots(doc_id, start_date, end_date)
            all_available_slots.extend(slots)
        
        # Sort slots by datetime
        all_available_slots.sort(key=lambda x: x['datetime'])
        
        return all_available_slots if all_available_slots else [{"message": "No available appointments found in the requested period"}]
        
    except Exception as e:
        return [{"error": f"Error checking appointment availability: {str(e)}"}]

@tool
def book_appointment(specialist_id: str, slot_datetime: str, customer_id: str, reason: str, thread_id: str) -> dict:
    """Book an appointment for a specific specialist for a customer at a specific time."""
    print ("\nbook_appointment called with: ", specialist_id, slot_datetime, customer_id, reason, thread_id)
    try:
        slot_dt = pd.to_datetime(slot_datetime)
        
        # Check if the slot is actually available (not during lunch, not weekend, etc.)
        if slot_dt.weekday() >= 5:  # Weekend
            return {"error": "Cannot book appointments on weekends"}
        
        if 14 <= slot_dt.hour < 15:  # Lunch break
            return {"error": "Cannot book appointments during lunch break (14:00-15:00)"}
        
        if not (11 <= slot_dt.hour < 18):  # Outside working hours
            return {"error": "Outside of working hours (11:00-18:00)"}
        
        # Check if slot is already booked
        existing_booking = appointments_df[
            (appointments_df['specialist_id'] == specialist_id) &
            (appointments_df['datetime'] == slot_dt) &
            (appointments_df['status'] == 'booked')
        ]
        
        if not existing_booking.empty:
            return {"error": "This time slot is already booked"}
        
        # Create new appointment
        new_appointment = {
            'appointment_id': f"APT-{len(appointments_df) + 1000}",
            'specialist_id': specialist_id,
            'customer_id': customer_id,
            'datetime': slot_dt.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_minutes': 30,
            'status': 'booked',
            'reason': reason,
            'thread_id': thread_id
        }
        
        # Add to dataframe
        appointments_df.loc[len(appointments_df)] = new_appointment
        
        # Save back to CSV
        appointments_df.to_csv('data/appointments.csv', index=False)
        
        return {
            "message": "Appointment booked successfully",
            "appointment_details": new_appointment
        }
        
    except Exception as e:
        return {"error": f"Error booking appointment: {str(e)}"}