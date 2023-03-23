from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

app = Flask(__name__)

# Load the machine learning model
model = joblib.load('line_breaking_pass_model.pkl')

# Load the soccer data into a pandas DataFrame
data = pd.read_csv('soccer_data.csv')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the player's ID and position
    player_id = int(request.form['player_id'])
    player_x = float(request.form['player_x'])
    player_y = float(request.form['player_y'])
    
    # Find the two nearest defenders
    defenders = data[data['Team'] != 'ball']  # Filter out non-defenders
    defenders['distance_to_player'] = np.sqrt((defenders['x'] - player_x)**2 + (defenders['y'] - player_y)**2)
    nearest_defenders = defenders.nsmallest(2, 'distance_to_player')[['x', 'y']].values
    
    # Get the weather data
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    wind_speed = float(request.form['wind_speed'])
    precipitation = float(request.form['precipitation'])
    
    # Make the prediction
    prediction = model.predict([[player_x, player_y, nearest_defenders[0
