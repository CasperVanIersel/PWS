
import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import time
import plotly.express as px
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import serial
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pymysql
from PIL import Image
from mysql.connector import connection
import mysql.connector

# Database instellingen
DB_PATH = "sensor_data.db"

# Database MYSQL connecten
db = mysql.connector.connect(host='localhost',
                             user='root',
                             password="theGrowGo2024Casper", # Jack ww: theGrowGo2024
                             database="growgo1",
                             auth_plugin = 'mysql_native_password' )

# Initialize serial connection
ser = None

# Initialiseer session state variabelen
if "import_running" not in st.session_state:
    st.session_state.import_running = False

if "start_time" not in st.session_state:
    st.session_state.start_time = int(datetime.now().timestamp())

if "time_offset" not in st.session_state:
    st.session_state.time_offset = 0

def initialize_database():
    """Initialiseer de database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dht11_1_temp REAL,
            dht11_1_hum REAL,
            dht11_2_temp REAL,
            dht11_2_hum REAL,
            dht11_3_temp REAL,
            dht11_3_hum REAL,
            soil_1 REAL,
            soil_2 REAL,
            soil_3 REAL,
            light_1 REAL,
            light_2 REAL,
            light_3 REAL
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS crop_sensor_mapping (
            crop_name TEXT PRIMARY KEY,
            dht11_sensor TEXT,
            soil_sensor TEXT,
            light_sensor TEXT,
            optimum_temp REAL,
            optimum_hum REAL,
            optimum_soil REAL,
            optimum_light REAL,
            avg_harvest_time REAL,
            avg_growth REAL,
            possible_diseases TEXT,
            disease_signs TEXT,
            water_needs REAL
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS plant_status (
            timestamp INTEGER PRIMARY KEY,
            crop_name TEXT,
            status TEXT,
            length REAL
        )
        """)

        # Check and add missing columns
        cursor.execute("PRAGMA table_info(crop_sensor_mapping)")
        columns = [info[1] for info in cursor.fetchall()]
        required_columns = {
            "optimum_temp": "REAL",
            "optimum_hum": "REAL",
            "optimum_soil": "REAL",
            "optimum_light": "REAL",
            "avg_harvest_time": "REAL",
            "avg_growth": "REAL",
            "possible_diseases": "TEXT",
            "disease_signs": "TEXT",
            "water_needs": "REAL"
        }
        for column, col_type in required_columns.items():
            if column not in columns:
                cursor.execute(f"ALTER TABLE crop_sensor_mapping ADD COLUMN {column} {col_type}")
    conn.close()

def save_data_to_db(data):
    """Opslaan van sensor data in de database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO sensor_data (
            dht11_1_temp, dht11_1_hum,
            dht11_2_temp, dht11_2_hum,
            dht11_3_temp, dht11_3_hum,
            soil_1, soil_2, soil_3,
            light_1, light_2, light_3
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        conn.commit()

def parse_arduino_data(line):
    """Parse a line of data from the Arduino."""
    try:
        parts = line.strip().split('|')
        temp1, hum1 = map(float, parts[1].split(':')[1].split(','))
        temp2, hum2 = map(float, parts[2].split(':')[1].split(','))
        temp3, hum3 = map(float, parts[3].split(':')[1].split(','))
        soil1 = int(parts[4].split(':')[1])
        soil2 = int(parts[5].split(':')[1])
        soil3 = int(parts[6].split(':')[1])  # Moisture sensor 3
        ldr1 = int(parts[7].split(':')[1])
        ldr2 = int(parts[8].split(':')[1])
        ldr3 = int(parts[9].split(':')[1])
        return (
            temp1, hum1, temp2, hum2, temp3, hum3,
            soil1, soil2, soil3, ldr1, ldr2, ldr3
        )
    except (IndexError, ValueError) as e:
        st.error(f"Data format is incorrect: {line}")
        return None

def read_arduino_data():
    """Lees data van Arduino."""
    global ser
    if ser is None:
        st.error("Arduino is not connected!")
        return None

    try:
        line = ser.readline().decode('utf-8').strip()
        data = parse_arduino_data(line)
        if data is None:
            return None
        return data
    except Exception as e:
        st.error(f"Failed to read data from Arduino: {e}")
        return None

def add_crop_to_db(crop_data):
    """Voeg een gewas toe aan de database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO crop_sensor_mapping (
            crop_name, dht11_sensor, soil_sensor, light_sensor,
            optimum_temp, optimum_hum, optimum_soil, optimum_light,
            avg_harvest_time, avg_growth, possible_diseases, disease_signs, water_needs
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, crop_data)
        conn.commit()

def delete_crop_from_db(crop_name):
    """Verwijder een gewas uit de database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM crop_sensor_mapping WHERE crop_name = ?", (crop_name,))
        cursor.execute("DELETE FROM plant_status WHERE crop_name = ?", (crop_name,))
        conn.commit()

def save_plant_status(status_data):
    """Opslaan van plant status in de database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO plant_status (
            timestamp, crop_name, status, length
        )
        VALUES (?, ?, ?, ?)
        """, status_data)
        conn.commit()

class GrowthModel(nn.Module):
    def __init__(self):
        super(GrowthModel, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def predict_growth(crop_name):
    """Voorspel de groei van een gewas."""
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(f"SELECT * FROM plant_status WHERE crop_name = '{crop_name}'", conn)

    if df.empty:
        return "Geen gegevens beschikbaar voor voorspelling."

    X = df[['timestamp']].values
    y = df['length'].values

    # Normalize the data
    X = (X - X.min()) / (X.max() - X.min())
    y = (y - y.min()) / (y.max() - y.min())

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Create and train the model
    model = GrowthModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    # Predict future growth for the next 4 months
    predictions = []
    for days in range(0, 120, 30):  # Predict every 30 days for 4 months
        future_timestamp = int(datetime.now().timestamp()) + (days * 24 * 60 * 60)
        future_timestamp_normalized = (future_timestamp - X.min()) / (X.max() - X.min())
        future_timestamp_tensor = torch.tensor([[future_timestamp_normalized]], dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            predicted_length_normalized = model(future_timestamp_tensor)
        predicted_length = predicted_length_normalized * (y.max() - y.min()) + y.min()
        predictions.append((future_timestamp, predicted_length.item()))

    return predictions

def plot_growth(crop_name, predictions):
    """Plot de groei van een gewas."""
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(f"SELECT * FROM plant_status WHERE crop_name = '{crop_name}'", conn)

    if df.empty:
        st.warning("Geen gegevens beschikbaar voor plotten.")
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        future_df = pd.DataFrame(predictions, columns=['timestamp', 'length'])
        future_df['timestamp'] = pd.to_datetime(future_df['timestamp'], unit='s')
        combined_df = pd.concat([df, future_df])
        fig = px.line(combined_df, x='timestamp', y='length', title=f'Groei van {crop_name}', labels={'timestamp': 'Tijd', 'length': 'Lengte (cm)'})
        st.plotly_chart(fig)

def populate_database_with_crops():
    """Populeer de database met realistische gewasgegevens."""
    crops = [
        ("Tomato", "dht11_1", "soil_1", "light_1", 20.0, 60.0, 500.0, 600.0, 90, 0.5, "Blight, Fusarium Wilt",
         "Yellowing leaves, wilting", 1.5),
        ("Lettuce", "dht11_2", "soil_2", "light_2", 18.0, 70.0, 400.0, 500.0, 45, 0.3, "Downy Mildew, Aphids",
         "White spots, curling leaves", 1.0),
        ("Cucumber", "dht11_3", "soil_3", "light_3", 25.0, 80.0, 600.0, 700.0, 60, 0.4,
         "Powdery Mildew, Cucumber Beetles", "White powdery spots, holes in leaves", 2.0),
        ("Strawberry", "dht11_1", "soil_1", "light_1", 22.0, 65.0, 450.0, 550.0, 75, 0.2, "Gray Mold, Spider Mites",
         "Gray fuzzy mold, tiny webs", 1.2),
        ("Bell Pepper", "dht11_2", "soil_2", "light_2", 24.0, 70.0, 500.0, 600.0, 80, 0.3, "Anthracnose, Aphids",
         "Dark sunken lesions, sticky leaves", 1.8)
    ]
    for crop in crops:
        add_crop_to_db(crop)

def analyze_description(description):
    """Analyze the description of the plant status."""
    positive_keywords = ["green", "healthy", "growing", "strong"]
    negative_keywords = ["brown", "wilted", "dying", "weak"]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([description] + positive_keywords + negative_keywords)
    description_vector = vectors[0]
    positive_vectors = vectors[1:len(positive_keywords) + 1]
    negative_vectors = vectors[len(positive_keywords) + 1:]

    positive_similarity = cosine_similarity(description_vector, positive_vectors).mean()
    negative_similarity = cosine_similarity(description_vector, negative_vectors).mean()

    if positive_similarity > negative_similarity:
        return "positive"
    else:
        return "negative"

# Streamlit-configuratie
st.set_page_config(layout="wide")

# Initialiseer database
initialize_database()
populate_database_with_crops()

# Sidebar navigatie
page = st.sidebar.selectbox("Navigatie", ["Home", "Data Management", "Grafieken", "Sensor-Gewas Koppeling", "AI Voorspelling", "AI Uitleg", "Progression and prospects"])

language = st.sidebar.selectbox('Choose your language / Kies je taal', ('English', 'Nederlands'))

st.sidebar.image("images/Growgo2.png", use_column_width=True)

# Terug naar boven knop met HTML en JavaScript
back_to_top_button = """
    <a href="#top" style="
        position: fixed;
        bottom: 50px;
        right: 50px;
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        border-radius: 5px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        font-size: 14px;
        z-index: 1000;
    ">↑ Terug naar boven</a>
"""

st.markdown('<div id="top"></div>', unsafe_allow_html=True)  # Plaats een anker bovenaan

if page == "Home":
    # App openen
    try:
        if language == 'English':
            st.title("The GrowGo Response Unit")
            st.header("Our new way to emergency aid for a world with less hunger")
            st.write(
                'Welcome to GrowGo, home of the innovative GrowGo Response Unit (GRU). Designed to provide fast, reliable access to emergency food supplies, the GRU is a portable, self-sustaining unit that allows you to grow crops quickly in any environment. Whether facing natural disasters, food shortages, or challenging climates, our GRU is engineered to ensure fresh, nutritious food is always within reach. With easy setup and rapid growth capabilities, GrowGo is your trusted partner in food security, delivering resilience when you need it most.')
            image = Image.open('images/AI growgo.jpg')
            st.image(image, caption="AI Generated", use_column_width=True)

        if language == 'Nederlands':
            st.title("De GrowGo Response Unit")
            st.header("Onze nieuwe weg om noodhulp te bieden voor een wereld met minder honger")
            st.write(
                'Welkom bij GrowGo, de thuisbasis van de innovatieve GrowGo Response Unit (GRU). De GRU is ontworpen om snel en betrouwbaar toegang te bieden tot noodvoedselvoorraden. Het is een draagbare, zelfvoorzienende unit waarmee je snel gewassen kunt kweken in elke omgeving. Of je nu te maken hebt met natuurrampen, voedseltekorten of moeilijke klimaten, onze GRU is ontworpen om ervoor te zorgen dat vers, voedzaam voedsel altijd binnen handbereik is. Dankzij de eenvoudige installatie en snelle groeimogelijkheden is GrowGo je betrouwbare partner in voedselzekerheid, en biedt het weerstand wanneer je het het meest nodig hebt.')
            image = Image.open('images/AI growgo.jpg')
            st.image(image, caption="AI Gegenereerd", use_column_width=True)
    except:
        if language == 'English':
            print("Error 101, contact website builders")
        if language == 'Nederlands':
            print("Error 101, neem contact op met de website ontwikkelaars")

if page == "Data Management":
    st.title("Data Management")
    col1, col2, col3 = st.columns(3)

    with col1:
        start_import = st.button("Start Importeren")
    with col2:
        stop_import = st.button("Stop Importeren")
    with col3:
        clear_db = st.button("Leeg Database")

    # Placeholder voor tabelweergave
    data_placeholder = st.empty()

    if clear_db:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM sensor_data")
        st.success("Database geleegd!")
        st.session_state.time_offset = 0

    # Start importeren
    if start_import:
        if not st.session_state.import_running:
            st.session_state.import_running = True
            st.session_state.start_time = int(datetime.now().timestamp()) - st.session_state.time_offset

            # Connect to Arduino
            try:
                ser = serial.Serial('COM3', 9600)
                st.success("Connected to Arduino successfully!")
            except Exception as e:
                st.error(f"Failed to connect to Arduino: {e}")
                st.session_state.import_running = False

    # Stop importeren
    if stop_import:
        st.session_state.import_running = False
        st.session_state.time_offset += int(datetime.now().timestamp()) - st.session_state.start_time

    # Importeren van data
    if st.session_state.import_running:
        st.info("Data importeren gestart...")
        while st.session_state.import_running:
            new_data = read_arduino_data()
            if new_data:
                save_data_to_db(new_data)

            with sqlite3.connect(DB_PATH) as conn:
                db_data = pd.read_sql("SELECT * FROM sensor_data ORDER BY id DESC", conn)
            data_placeholder.dataframe(db_data)
            time.sleep(2)

    st.title("Inspect Database")
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM sensor_data", conn)
    st.write(df)

elif page == "Grafieken":
    st.title("Sensor Grafieken")
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM sensor_data", conn)

    if df.empty:
        st.warning("Geen gegevens beschikbaar.")
    else:
        sensor_columns = [col for col in df.columns if col not in ["id"]]
        selected_sensors = st.multiselect("Selecteer sensoren", sensor_columns, sensor_columns)
        if selected_sensors:
            filtered_data = df.melt(id_vars=["id"], value_vars=selected_sensors)
            fig = px.line(
                filtered_data,
                x="id",
                y="value",
                color="variable",
                title="Sensor Waarden",
                labels={"id": "ID", "value": "Sensorwaarde"}
            )
            st.plotly_chart(fig)

elif page == "Sensor-Gewas Koppeling":
    st.title("Sensor-Gewas Koppeling")
    crop_name = st.text_input("Gewasnaam")
    dht11_sensor = st.selectbox("DHT11 Sensor", ["dht11_1", "dht11_2", "dht11_3"])
    soil_sensor = st.selectbox("Bodemvochtigheid Sensor", ["soil_1", "soil_2", "soil_3"])
    light_sensor = st.selectbox("Licht Sensor", ["light_1", "light_2", "light_3"])
    optimum_temp = st.number_input("Optimale Temperatuur (°C)", min_value=0.0, max_value=50.0, step=0.1)
    optimum_hum = st.number_input("Optimale Vochtigheid (%)", min_value=0.0, max_value=100.0, step=0.1)
    optimum_soil = st.number_input("Optimale Bodemvochtigheid (units)", min_value=0.0, max_value=1000.0, step=1.0)
    optimum_light = st.number_input("Optimale Lichtintensiteit (units)", min_value=0.0, max_value=1000.0, step=1.0)
    avg_harvest_time = st.number_input("Gemiddelde Oogsttijd (dagen)", min_value=0.0, step=1.0)
    avg_growth = st.number_input("Gemiddelde Groei (cm/dag)", min_value=0.0, step=0.1)
    possible_diseases = st.text_area("Mogelijke Ziekten")
    disease_signs = st.text_area("Ziekte Symptomen")
    water_needs = st.number_input("Waterbehoefte (L/dag)", min_value=0.0, step=0.1)

    if st.button("Voeg Gewas Toe"):
        crop_data = (
            crop_name, dht11_sensor, soil_sensor, light_sensor,
            optimum_temp, optimum_hum, optimum_soil, optimum_light,
            avg_harvest_time, avg_growth, possible_diseases, disease_signs, water_needs
        )
        add_crop_to_db(crop_data)
        st.success("Gewas toegevoegd aan de database!")

    crop_to_delete = st.selectbox("Selecteer Gewas om te Verwijderen", [row[0] for row in sqlite3.connect(DB_PATH).execute(
        "SELECT crop_name FROM crop_sensor_mapping").fetchall()])
    if st.button("Verwijder Gewas"):
        delete_crop_from_db(crop_to_delete)
        st.success("Gewas verwijderd uit de database!")

elif page == "AI Voorspelling":
    st.title("AI Voorspelling")
    crop_name = st.selectbox("Selecteer Gewas", [row[0] for row in sqlite3.connect(DB_PATH).execute(
        "SELECT crop_name FROM crop_sensor_mapping").fetchall()])
    status = st.text_area("Beschrijving van de huidige status van de plant")
    length = st.number_input("Huidige lengte van de plant (cm)", min_value=0.0, step=0.1)
    date = st.date_input("Datum van meting", datetime.now().date())

    if st.button("Sla Plant Status Op"):
        timestamp = int(datetime.combine(date, datetime.min.time()).timestamp())

        status_data = (timestamp, crop_name, status, length)

    if st.button("Voorspel Groei"):
        prediction = predict_growth(crop_name)

        st.info(prediction)
    if st.button("Plot Groei"):

        predictions = predict_growth(crop_name)  # Get predictions again if needed

        plot_growth(crop_name, predictions)

elif page == "Progression and prospects":

        def delete_crop_from_db(crop_name):
            try:
                # Maak een verbinding met de database
                conn = create_connection()
                cursor = conn.cursor()

                # SQL-query om het gewas te verwijderen
                sql = "DELETE FROM crops WHERE name = %s"
                cursor.execute(sql, (crop_name,))
                conn.commit()  # Bevestig de verwijdering

                cursor.close()
                conn.close()

                if language == "English":
                    st.success(f"Crop '{crop_name}' has been successfully deleted.")
                if language == "Nederlands":
                    st.success(f"Gewas '{crop_name}' is succesvol verwijderd.")
            except mysql.connector.Error as err:
                if language == "English":
                    st.error(f"An error occurred while deleting the crop: {err}")
                if language == "Nederlands":
                    st.error(f"Fout bij het verwijderen van het gewas: {err}")


        def display_delete_crop_option():
            # Maak verbinding met de database om alle gewassen op te halen
            crops = get_all_crops()  # Deze functie zou alle gewassen uit de database moeten halen
            crop_names = [crop[1] for crop in crops]  # Neem alleen de namen van de gewassen

            if language == "English":
                st.header("Remove a crop")
            if language == "Nederlands":
                st.header("Verwijder een gewas")

            # Dropdown om het gewas te selecteren
            crop_to_delete = st.selectbox("Select a crop to delete", crop_names)

            # Verwijderknop
            if st.button("Delete Crop" if language == "English" else "Verwijder gewas"):
                delete_crop_from_db(crop_to_delete)
                st.rerun()  # Herlaad de pagina om de lijst te verversen

        # Functie om een verbinding met de MySQL-database te maken
        def create_connection():
            connection = pymysql.connect(
                host="localhost",  # Pas dit aan naar jouw MySQL host
                user="root",  # MySQL-gebruikersnaam
                password="theGrowGo2024Casper",  # MySQL-wachtwoord
                database="growgo1"  # De database die je eerder hebt aangemaakt
            )
            return connection


        # Functie om gegevens in te voegen in de MySQL-database
        def insert_crop_to_db(crop_name, optimum_temp, optimum_humidity, planting_date, grow_days):
            try:
                # Maak een verbinding met de database
                conn = create_connection()
                cursor = conn.cursor()

                # SQL-query om gegevens in de tabel "crops" in te voegen
                sql = """
                INSERT INTO crops (name, optimum_temp, optimum_humidity, planting_date, grow_days)
                VALUES (%s, %s, %s, %s, %s)
                """
                values = (crop_name, optimum_temp, optimum_humidity, planting_date, grow_days)

                # Print de query en waarden voor debugging
                print(f"Query: {sql}")
                print(f"Values: {values}")

                # Voer de query uit
                cursor.execute(sql, values)
                conn.commit()  # Zorgt ervoor dat de wijzigingen worden doorgevoerd

                # Sluit de cursor en verbinding netjes af
                cursor.close()
                conn.close()
                if language == "English":
                    print("Crop successfully added to database.")  # Debugging informatie
                if language == "Nederlands":
                    print("Gewas succesvol toegevoegd aan de database.")  # Debugging informatie

            except mysql.connector.Error as err:
                # Indien er een fout optreedt, geef de foutmelding weer
                if language == "English":
                    print(f"An error occurred while adding the Crop: {err}")
                if language == "Nederlands":
                    print(f"Fout bij het toevoegen van gewas: {err}")


        # Functie om alle gewassen uit de database te halen
        def get_all_crops():
            conn = create_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM crops")
            crops = cursor.fetchall()

            cursor.close()
            conn.close()

            return crops


        def calculate_age(planting_date):
            today = datetime.now().date()  # Haal alleen de datum op (date-object)
            age = (today - planting_date).days  # Bereken de leeftijd in dagen
            return age





        # Functie om de geschatte oogstdatum te berekenen
        def calculate_harvest_date(planting_date, grow_days):
            return planting_date + timedelta(days=grow_days)


        def display_add_crop_form():
            if language == 'English':
                st.header("Add new crops")
            if language == 'Nederlands':
                st.header("Voeg nieuwe gewassen toe")

            # Formulier voor het toevoegen van een nieuw gewas
            with st.form("new_plant_form"):
                crop_name = st.text_input("Naam van het gewas" if language == 'Nederlands' else "Crop name")
                optimum_temp = st.number_input(
                    "Optimale temperatuur (°C)" if language == 'Nederlands' else "Optimal temperature (°C)",
                    min_value=0, max_value=50)
                optimum_humidity = st.number_input(
                    "Optimale vochtigheid (%)" if language == 'Nederlands' else "Optimal humidity (%)", min_value=0,
                    max_value=100)
                planting_date = st.date_input("Plantdatum" if language == 'Nederlands' else "Planting date",
                                              datetime.now())
                grow_days = st.number_input(
                    "Geschatte groei in dagen" if language == 'Nederlands' else "Estimated grow in days", min_value=1,
                    max_value=365)

                submitted = st.form_submit_button("Voeg gewas toe" if language == 'Nederlands' else "Add new crops")

                if submitted:
                    age = calculate_age(planting_date)  # planting_date komt al als een date-object
                    harvest_date = calculate_harvest_date(planting_date, grow_days)

                    # Voeg het gewas toe aan de database
                    insert_crop_to_db(crop_name, optimum_temp, optimum_humidity, planting_date.strftime('%Y-%m-%d'),
                                      grow_days)
                    st.success(
                        f"{crop_name} toegevoegd. Leeftijd: {age} dagen. Verwachte oogstdatum: {harvest_date.strftime('%Y-%m-%d')}" if language == 'Nederlands' else f"{crop_name} added. Age: {age} days. Expected harvest date: {harvest_date.strftime('%Y-%m-%d')}")


        # Functie om de lijst van gewassen weer te geven
        def display_crops():
            st.header("Crops List" if language == 'English' else "Lijst van gewassen")

            crops = get_all_crops()

            with st.expander("Show list of crops" if language == 'English' else "Bekijk gewassenlijst"):
                if crops:
                 for crop in crops:
                    planting_date = crop[4]  # plantdatum in de juiste indeling
                    age = calculate_age(planting_date)
                    harvest_date = calculate_harvest_date(planting_date, crop[5])  # crop[5] is grow_days
                    st.write(f"Crop: {crop[1]}" if language == 'English' else f"Gewas: {crop[1]}")  # crop[1] is naam
                    st.write(f"Age: {age} days" if language == 'English' else f"Leeftijd: {age} dagen")
                    st.write(
                        f"Expected harvest date: {harvest_date.strftime('%Y-%m-%d')}" if language == 'English' else f"Verwachte oogstdatum: {harvest_date.strftime('%Y-%m-%d')}")
                    st.write("---")
                else:
                    st.write("No crops found." if language == 'English' else "Geen gewassen gevonden.")



        if language == "English":
            st.header("Manage crops")
        if language == "Nederlands":
            st.header("Beheer gewassen")

        # Toon het formulier voor het toevoegen van nieuwe gewassen
        display_add_crop_form()  # Deze functie heb je al voor het toevoegen van gewassen

        # Toont de lijst van gewassen
        display_crops()

        # Toon de optie om gewassen te verwijderen
        display_delete_crop_option()  # Voeg de optie voor verwijderen toe

        # Knop voor composthoeveelheid
        if language == 'English':
            st.header("Compost quantity and estimate")
        else:
            st.header("Composthoeveelheid en schatting")

        # Formulier voor composthoeveelheid
        with st.form("compost_form"):
            current_compost = st.number_input(
                "Current compost quantity (kg)" if language == 'English' else "Huidige hoeveelheid compost (kg)",
                min_value=0)
            compost_needed = st.number_input(
                "Compost needed for future crops (kg)" if language == 'English' else "Benodigde compost voor toekomstige gewassen (kg)",
                min_value=0)

            compost_submitted = st.form_submit_button(
                "Estimate compost quantity" if language == 'English' else "Schat composthoeveelheid")

            if compost_submitted:
                compost_diff = current_compost - compost_needed
                if compost_diff >= 0:
                    st.success(
                        f"You have enough compost. Surplus: {compost_diff} kg." if language == 'English' else f"Je hebt voldoende compost. Overschot: {compost_diff} kg.")
                else:
                    st.warning(
                        f"You need more compost! Deficit: {-compost_diff} kg." if language == 'English' else f"Je hebt meer compost nodig! Tekort: {-compost_diff} kg.")

            st.markdown(back_to_top_button, unsafe_allow_html=True)