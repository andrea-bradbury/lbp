from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

# Load the data and model
data = pd.read_csv('soccer_data.csv')
model = pickle.load(open('line_breaking_pass_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the player's ID, x and y coordinates, and weather data from the web form
    player_id = request.form['player_id']
    player_x = float(request.form['player_x'])
    player_y = float(request.form['player_y'])
    defender_x1 = float(request.form['defender_x1'])
    defender_y1 = float(request.form['defender_y1'])
    defender_x2 = float(request.form['defender_x2'])
    defender_y2 = float(request.form['defender_y2'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    wind_speed = float(request.form['wind_speed'])
    precipitation = float(request.form['precipitation'])

    # Calculate the distance between the player and the defenders
    defender_distances = np.sqrt((data['x'] - player_x)**2 + (data['y'] - player_y)**2)
    closest_defenders = data.loc[defender_distances.nsmallest(2).index]

    # Make a prediction using the model
    features = np.array([
        player_x, player_y,
        closest_defenders.iloc[0]['x'], closest_defenders.iloc[0]['y'],
        closest_defenders.iloc[1]['x'], closest_defenders.iloc[1]['y'],
        temperature, humidity, wind_speed, precipitation
    ]).reshape(1, -1)
    prediction = model.predict(features)[0]

    # Render the prediction as a web page
    return render_template('prediction.html',
        player_id=player_id,
        player_x=player_x, player_y=player_y,
        defender_x1=defender_x1, defender_y1=defender_y1,
        defender_x2=defender_x2, defender_y2=defender_y2,
        temperature=temperature, humidity=humidity,
        wind_speed=wind_speed, precipitation=precipitation,
        prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
