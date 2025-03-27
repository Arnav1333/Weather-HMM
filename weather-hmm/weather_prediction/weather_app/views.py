from django.shortcuts import render
import numpy as np
import requests
from collections import defaultdict
from datetime import datetime, timedelta
import pytz
from django.shortcuts import render

# Weather mapping
weather_mapping = {
    "Clear": "Sunny",
    "Clouds": "Cloudy",
    "Rain": "Rainy",
    "Drizzle": "Rainy",
    "Thunderstorm": "Rainy",
    "Snow": "Rainy",
    "Mist": "Cloudy",
    "Smoke": "Cloudy",
    "Haze": "Cloudy",
    "Dust": "Cloudy",
    "Fog": "Cloudy",
    "Sand": "Cloudy",
    "Ash": "Cloudy",
    "Squall": "Rainy",
    "Tornado": "Rainy"
}

def map_weather_description(description):
    for key, value in weather_mapping.items():
        if key.lower() in description.lower():
            return value
    return "Sunny"  # Default if no match

def estimate_hmm_probabilities(historical_weather_states, states):
    """Train Hidden Markov Model for weather prediction"""
    n_states = len(states)
    initial_counts = defaultdict(int)
    transition_counts = defaultdict(lambda: defaultdict(int))

    if not historical_weather_states:
        return np.array([1/n_states] * n_states), np.full((n_states, n_states), 1/n_states)

    initial_counts[historical_weather_states[0]] += 1

    for i in range(len(historical_weather_states) - 1):
        current_state = historical_weather_states[i]
        next_state = historical_weather_states[i+1]
        transition_counts[current_state][next_state] += 1

    initial_probs = np.array([initial_counts[state] / len(historical_weather_states)
                              if len(historical_weather_states) > 0 else 1/n_states
                              for state in states])

    transition_probs = np.zeros((n_states, n_states))
    for i, from_state in enumerate(states):
        total_transitions = sum(transition_counts[from_state].values())
        if total_transitions > 0:
            for j, to_state in enumerate(states):
                transition_probs[i, j] = transition_counts[from_state].get(to_state, 0) / total_transitions
        else:
            transition_probs[i, :] = 1 / n_states  

    return initial_probs, transition_probs

def predict_next_weather_hmm(last_state, states, transition_probs):
    """Predict next weather state using HMM"""
    last_state_index = states.index(last_state)
    next_state_probs = transition_probs[last_state_index]
    next_state_index = np.random.choice(len(next_state_probs), p=next_state_probs)
    return states[next_state_index]

def weather_forecast(request):
    forecast = None
    city = None

    if request.method == "POST":
        city = request.POST.get("city", "").strip()
        if city:
            api_key = "QEG4GG8KYM766GKKSJNW5N2X9"
            historical_data_url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}?unitGroup=us&key={api_key}&contentType=json"

            try:
                response = requests.get(historical_data_url)
                response.raise_for_status()
                data = response.json()

                historical_weather_states = []
                if 'days' in data:
                    for day_data in data['days']:
                        description = day_data.get('description', '')
                        weather_state = map_weather_description(description)
                        historical_weather_states.append(weather_state)

                states = ["Sunny", "Cloudy", "Rainy"]
                if historical_weather_states:
                    _, transition_probs_trained = estimate_hmm_probabilities(historical_weather_states, states)

                    current_weather_description = data.get('currentConditions', {}).get('conditions', None)
                    if current_weather_description:
                        current_weather = map_weather_description(current_weather_description)

                        forecast = []
                        last_predicted_state = current_weather
                        ist = pytz.timezone('Asia/Kolkata')
                        now_ist = datetime.now(ist).date()

                        for i in range(3):
                            next_predicted_state = predict_next_weather_hmm(last_predicted_state, states, transition_probs_trained)
                            last_predicted_state = next_predicted_state
                            future_date = now_ist + timedelta(days=i + 1)
                            forecast.append((future_date.strftime('%Y-%m-%d'), next_predicted_state))
                else:
                    forecast = "No historical data found."

            except requests.exceptions.RequestException:
                forecast = "Error fetching weather data."

    return render(request, "index.html", {"city": city, "forecast": forecast})
