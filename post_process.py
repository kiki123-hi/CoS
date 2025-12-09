import numpy as np
from math import radians, sin, cos, sqrt, asin
from typing import List, Tuple, Optional, Dict
import time
import json
from datetime import datetime, timedelta


class Event:
    def __init__(self, start_time_str: str, end_time_str: str, score: float,
                 lat: float, lon: float, id: str = "", original_index: int = -1):
        self.start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
        self.end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")

        self.score = score if isinstance(score, (int, float)) and not (score != score) else 0.0
        self.lat = lat if isinstance(lat, (int, float)) and not (lat != lat) else 0.0
        self.lon = lon if isinstance(lon, (int, float)) and not (lon != lon) else 0.0
        self.id = id or f"Event_{start_time_str}-{end_time_str}"
        self.original_index = original_index

    def __repr__(self):
        return (f"Event(id={self.id}, score={self.score}, loc=({self.lat:.4f}, {self.lon:.4f}), "
                f"time={self.start_time.strftime('%Y-%m-%d %H:%M')}-{self.end_time.strftime('%H:%M')})")


def haversine_distance(event1: Event, event2: Event) -> float:
    R = 6371.0

    lat1, lon1, lat2, lon2 = map(radians, [event1.lat, event1.lon, event2.lat, event2.lon])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    return R * c


def check_time_conflict(event1: Event, event2: Event) -> bool:
    return not (event1.end_time < event2.start_time or event2.end_time < event1.start_time)


global_alternative_events_pool: List[Event] = []


def find_optimal_plan_with_alternatives(
        main_events: List[Event],
        travel_speed_km_per_min: float
) -> Tuple[float, List[Event]]:
    n = len(main_events)
    if n == 0:
        return 0.0, []

    dp = [0.0] * n
    prev = [-1] * n
    actual_selected_event = [None] * n

    max_score = 0.0
    max_index = -1
    if n > 0:
        max_score = max(e.score for e in main_events)
        max_index = [e.score for e in main_events].index(max_score)

    for i in range(n):
        current_main_event = main_events[i]
        dp[i] = current_main_event.score
        actual_selected_event[i] = current_main_event

        if dp[i] > max_score:
            max_score = dp[i]
            max_index = i

    for i in range(n):
        current_main_event = main_events[i]

        for j in range(i):
            prev_event_in_sequence = actual_selected_event[j]

            if prev_event_in_sequence is None:
                continue

            if check_time_conflict(prev_event_in_sequence, current_main_event):
                continue

            distance_constraint_met = True
            event_to_use_in_sequence = current_main_event

            time_diff_minutes = (current_main_event.start_time - prev_event_in_sequence.end_time).total_seconds() / 60

            dynamic_max_distance = time_diff_minutes * travel_speed_km_per_min
            actual_dist = haversine_distance(prev_event_in_sequence, current_main_event)

            if actual_dist > dynamic_max_distance:
                distance_constraint_met = False

                eligible_alternatives = []
                for alt_event in global_alternative_events_pool:
                    if check_time_conflict(prev_event_in_sequence, alt_event):
                        continue

                    alt_time_diff_minutes = (
                                                        alt_event.start_time - prev_event_in_sequence.end_time).total_seconds() / 60
                    if alt_time_diff_minutes < 0:
                        continue

                    alt_dynamic_max_distance = alt_time_diff_minutes * travel_speed_km_per_min
                    alt_dist = haversine_distance(prev_event_in_sequence, alt_event)

                    if alt_event.score is None:
                        continue

                    if alt_dist <= alt_dynamic_max_distance:
                        eligible_alternatives.append(alt_event)

                eligible_alternatives.sort(key=lambda x: x.score, reverse=True)

                if eligible_alternatives:
                    selected_alternative = eligible_alternatives[0]
                    distance_constraint_met = True
                    event_to_use_in_sequence = selected_alternative
                else:
                    distance_constraint_met = False

            if distance_constraint_met:
                current_event_score = event_to_use_in_sequence.score
                new_score = dp[j] + current_event_score

                if new_score > dp[i]:
                    dp[i] = new_score
                    prev[i] = j
                    actual_selected_event[i] = event_to_use_in_sequence

        if dp[i] > max_score:
            max_score = dp[i]
            max_index = i

    sequence = []
    current_idx = max_index
    while current_idx != -1:
        if actual_selected_event[current_idx]:
            sequence.append(actual_selected_event[current_idx])
        current_idx = prev[current_idx]
    sequence.reverse()

    return max_score, sequence


def validate_sequence(sequence: List[Event], travel_speed_km_per_min: float) -> bool:
    if len(sequence) <= 1:
        return True

    for i in range(1, len(sequence)):
        prev_event = sequence[i - 1]
        current_event = sequence[i]

        if check_time_conflict(prev_event, current_event):
            print(
                f"Validation failed: Time conflict! {prev_event.id} (ends at {prev_event.end_time.strftime('%Y-%m-%d %H:%M')}) with {current_event.id} (starts at {current_event.start_time.strftime('%Y-%m-%d %H:%M')})")
            return False

        time_diff_minutes = (current_event.start_time - prev_event.end_time).total_seconds() / 60

        if time_diff_minutes < 0:
            print(f"Validation failed: Logical error, event time overlap but not caught by check_time_conflict.")
            return False

        dynamic_max_distance = time_diff_minutes * travel_speed_km_per_min
        actual_dist = haversine_distance(prev_event, current_event)

        if actual_dist > dynamic_max_distance:
            print(
                f"Validation failed: Distance constraint violated! {prev_event.id} to {current_event.id} = {actual_dist:.2f}km (Allowed: {dynamic_max_distance:.2f}km, Travel Time: {time_diff_minutes:.2f} minutes)")
            return False

    return True


def calculate_total_utility(schedule_events: List[Event]) -> float:
    valid_utilities = [event.score for event in schedule_events]
    return sum(valid_utilities)


def calculate_average_utility(schedule_events: List[Event]) -> float:
    valid_utilities = [event.score for event in schedule_events]
    if not valid_utilities:
        return 0.0
    return sum(valid_utilities) / len(valid_utilities)


def run_full_simulation(file_path: Optional[str] = None):
    global global_alternative_events_pool

    all_schedules_data = []
    if file_path:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                all_schedules_data = json.load(f)

            if not (isinstance(all_schedules_data, list) and len(all_schedules_data) > 0):
                print("Error: JSON file is empty or not in the expected format (a list of dictionaries).")
                all_schedules_data = []
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
            all_schedules_data = []
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from the file '{file_path}'. Check for valid JSON format.")
            all_schedules_data = []
        except Exception as e:
            print(f"An unexpected error occurred during file loading: {e}")
            all_schedules_data = []
    else:
        print("No file path provided. Exiting simulation. Please provide a JSON file path to run the simulation.")
        return

    planning_times = []
    all_processed_schedules_for_metrics: List[List[Event]] = []

    travel_speed_km_per_min = 0.5

    total_algorithm_start_time = time.perf_counter()

    if not all_schedules_data:
        print("No valid data to process. Exiting simulation.")
        return

    for entry_index, entry in enumerate(all_schedules_data):
        print(f"\n--- Processing Scenario {entry_index + 1} ---")

        iteration_start_time = time.perf_counter()

        main_event_list_raw = entry.get("Schedule", [])
        global_alternative_pool_raw = entry.get("Input", [])

        main_events = [
            Event(
                e.get("Start Time"),
                e.get("End Time"),
                e.get("Utility Score"),
                e.get("Latitude"),
                e.get("Longitude"),
                e.get("Event Id"),
                idx
            ) for idx, e in enumerate(main_event_list_raw)
        ]

        global_alternative_events_pool.clear()
        global_alternative_events_pool.extend([
            Event(
                e.get("Start Time"),
                e.get("End Time"),
                e.get("Utility Score"),
                e.get("Latitude"),
                e.get("Longitude"),
                e.get("Event Id"),
                -1
            ) for e in global_alternative_pool_raw
        ])

        print("--- Main Events for this scenario ---")
        for event in main_events:
            print(event)

        print("\n--- Alternative Events for this scenario ---")
        for event in global_alternative_events_pool:
            print(event)

        max_score, optimal_sequence = find_optimal_plan_with_alternatives(main_events, travel_speed_km_per_min)

        iteration_end_time = time.perf_counter()
        planning_time = iteration_end_time - iteration_start_time
        planning_times.append(planning_time)
        all_processed_schedules_for_metrics.append(optimal_sequence)

        current_total_utility = calculate_total_utility(optimal_sequence)
        current_average_utility = calculate_average_utility(optimal_sequence)

        print("\n--- Optimal Event Plan for Current Scenario ---")
        if not optimal_sequence:
            print("No valid sequence found for this scenario.")
        else:
            prev_event_for_dist: Optional[Event] = None
            for i, event in enumerate(optimal_sequence):
                distance_details = ""
                if prev_event_for_dist is not None:
                    actual_dist = haversine_distance(prev_event_for_dist, event)
                    time_diff_minutes_for_print = (event.start_time - prev_event_for_dist.end_time).total_seconds() / 60
                    allowed_dist_for_print = time_diff_minutes_for_print * travel_speed_km_per_min
                    distance_details = (f" | Dist from {prev_event_for_dist.id}: {actual_dist:.2f}km "
                                        f"(Allowed: {allowed_dist_for_print:.2f}km, Travel Time: {time_diff_minutes_for_print:.2f}min)")
                print(f"{i + 1}. {event}{distance_details}")
                prev_event_for_dist = event

        print(f"\nTotal Utility Score for this plan: {current_total_utility:.2f}")
        print(f"Average Utility Score per event in this plan: {current_average_utility:.2f}")
        print(f"Planning Time for this scenario: {planning_time:.4f} seconds")

        is_valid = validate_sequence(optimal_sequence, travel_speed_km_per_min)
        print(f"Sequence Validation Result: {'✅ All constraints met' if is_valid else '❌ Constraints violated'}")

    total_algorithm_end_time = time.perf_counter()
    overall_runtime = total_algorithm_end_time - total_algorithm_start_time

    print("\n" + "=" * 40)
    print("             Overall Algorithm Metrics")
    print("=" * 40)

    if planning_times:
        overall_total_utility = sum(calculate_total_utility(s) for s in all_processed_schedules_for_metrics)
        all_events_in_processed_schedules = [event for schedule in all_processed_schedules_for_metrics for event in
                                             schedule]
        overall_average_utility = calculate_average_utility(all_events_in_processed_schedules)

        print(f"Overall Algorithm Runtime: {overall_runtime:.4f} seconds")
        print(f"Number of Planning Scenarios Processed: {len(planning_times)}")
        print(f"Average Planning Time per Scenario: {sum(planning_times) / len(planning_times):.4f} seconds")
        print(f"Overall Total Utility Score of all planned events: {overall_total_utility:.2f}")
        print(f"Overall Average Utility Score of all planned events: {overall_average_utility:.2f}")
    else:
        print("No schedules were processed successfully.")


if __name__ == "__main__":
    file_path = "your_path"
    run_full_simulation(file_path)