# Bibliotheken
import serial
from mysql.connector import connection
import mysql.connector
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime, timedelta
import pymysql

# Database MYSQL connecten
db = mysql.connector.connect(host='localhost',
                             user='root',
                             password="theGrowGo2024Casper",
                             database="growgo1",
                             auth_plugin = 'mysql_native_password' )

cursor = db.cursor()

# Variabelen
x = 1  # variabele die verder in de code wordt gebruikt om errors te voorkomen

st.sidebar.header("Menu")  # dit is de titel van de sidebar
sideselectbox = st.sidebar.selectbox("Choose page",
                                     ["Home", "Plot Data", "Realtime plot and input", "Progression and prospects"],
                                     key="selectbox")
language = st.sidebar.selectbox('Choose your language / Kies je taal', ('English', 'Nederlands'))

# Datum een integer maken --> id voor iedere tabel is toch uniek
now = str(datetime.now())
timestamp_int1 = now.replace(".", "")
timestamp_int2 = timestamp_int1.replace("-", "")
timestamp_int3 = timestamp_int2.replace(":", "")
timestamp_int4 = timestamp_int3.replace(" ", "")
timestamp = int(timestamp_int4.replace(".", '')[0:14])
# print(timestamp)

# Deze lijsten zijn voor de RealtimePlot
# ArduinoAS
TemperatuurASList = []
AardeVochtASList = []
LuchtVochtASList = []
LichtASList = []

TemperatuurASdate = []
AardeVochtASdate = []
LuchtVochtASdate = []
LichtASdate = []

# Arduino1
TemperatuurArduino1Waarde1List = []
AardeVochtArduino1Waarde1List = []
LuchtVochtArduino1Waarde1List = []
LichtArduino1Waarde1List = []

TemperatuurArduino1Waarde1Date = []
AardeVochtArduino1Waarde1Date = []
LuchtVochtArduino1Waarde1Date = []
LichtArduino1Waarde1Date = []

TemperatuurArduino1Waarde2List = []
AardeVochtArduino1Waarde2List = []
LuchtVochtArduino1Waarde2List = []
LichtArduino1Waarde2List = []

TemperatuurArduino1Waarde2Date = []
AardeVochtArduino1Waarde2Date = []
LuchtVochtArduino1Waarde2Date = []
LichtArduino1Waarde2Date = []

# Arduino2
TemperatuurArduino2Waarde1List = []
AardeVochtArduino2Waarde1List = []
LuchtVochtArduino2Waarde1List = []
LichtArduino2Waarde1List = []

TemperatuurArduino2Waarde2List = []
AardeVochtArduino2Waarde2List = []
LuchtVochtArduino2Waarde2List = []
LichtArduino2Waarde2List = []

TemperatuurArduino2Waarde1Date = []
AardeVochtArduino2Waarde1Date = []
LuchtVochtArduino2Waarde1Date = []
LichtArduino2Waarde1Date = []

TemperatuurArduino2Waarde2Date = []
AardeVochtArduino2Waarde2Date = []
LuchtVochtArduino2Waarde2Date = []
LichtArduino2Waarde2Date = []


# Definities maken voor structuur in de code
def import_arduinoAS():  # dit is om alle data met specifieke begrippen in de MYSQL te doen, zelfde geldt voor de andere functies
    try:
        conn = mysql.connector.connect(db)  # handig voor later in de code
        if AardeVochtASwaarde != '':  # mocht er een error zijn
            cursor.execute("INSERT INTO AardeVochtAS (AardeVochtASwaarde, date) VALUES (%s, %s)",
                           # dit is eigenlijk gewoon een sql query
                           (AardeVochtASwaarde, timestamp))
            conn.commit()  # zorgt ervoor dat er echt verandering komt
            print("Aardevocht waarde is ", AardeVochtASwaarde)  # mocht er iets fout gaan

    except:
        print("Aardevocht error")  # mocht er iets fout zijn
        conn = mysql.connector.connect(db)

    try:
        if TemperatuurASwaarde != '':
            cursor.execute("INSERT INTO TemperatuurAS (TemperatuurASwaarde, date) VALUES (%s, %s)",
                           (TemperatuurASwaarde, timestamp))
            conn.commit()
            print("Temperatuur waarde is ", TemperatuurASwaarde)
    except:
        print('Temp error')
    try:
        if LuchtVochtASwaarde != '':
            cursor.execute("INSERT INTO LuchtVochtAS (LuchtVochtASwaarde, date) VALUES (%s, %s)",
                           (LuchtVochtASwaarde, timestamp))
            conn.commit()
            print("LuchtVocht is ", LuchtVochtASwaarde)
    except:
        print('LV error')
    try:
        if LichtASwaarde != '':
            cursor.execute("INSERT INTO LichtAS (LichtASwaarde, date) VALUES (%s, %s)",
                           (LichtASwaarde, timestamp))
            conn.commit()
            print("Licht waarde is ", LichtASwaarde)
    except:
        print('Licht error')


def import_arduino1():
    try:
        conn = mysql.connector.connect(db)  # handig voor later in de code
        if AardeVocht1arduino1_waarde != '':
            cursor.execute("INSERT INTO AardeVocht1arduino1 (AardeVocht1arduino1, date) VALUES (%s, %s)",
                           (AardeVocht1arduino1_waarde, timestamp))
            conn.commit()
            print("Aardevocht waarde is ", AardeVocht1arduino1_waarde)

    except:
        conn = mysql.connector.connect(db)
        print('AardeVocht error')

    try:
        if Temperatuur1arduino1_waarde != '':
            cursor.execute("INSERT INTO Temperatuur1arduino1 (Temperatuur1arduino1, date) VALUES (%s, %s)",
                           (Temperatuur1arduino1_waarde, timestamp))
            conn.commit()
            print("Temperatuur waarde is ", Temperatuur1arduino1_waarde)
    except:
        print('Temp error')
    try:
        if LuchtVocht1arduino1_waarde != '':
            cursor.execute("INSERT INTO LuchtVocht1arduino1 (LuchtVocht1arduino1_waarde, date) VALUES (%s, %s)",
                           (LuchtVocht1arduino1_waarde, timestamp))
            conn.commit()
            print("LuchtVocht is ", LuchtVocht1arduino1_waarde)
    except:
        print('LV error')
    try:
        if Licht1arduino1_waarde != '':
            cursor.execute("INSERT INTO Licht1arduino1 (Licht1arduino1_waarde, date) VALUES (%s, %s)",
                           (Licht1arduino1_waarde, timestamp))
            conn.commit()
            print("Licht waarde is ", Licht1arduino1_waarde)
    except:
        print('Licht error')
    try:
        if AardeVocht2arduino1_waarde != '':
            cursor.execute("INSERT INTO AardeVocht2arduino1 (AardeVocht2arduino1, date) VALUES (%s, %s)",
                           (AardeVocht2arduino1_waarde, timestamp))
            conn.commit()
            print("Aardevocht waarde is ", AardeVocht2arduino1_waarde)

    except:
        print('AardeVocht error')

    try:
        if Temperatuur2arduino1_waarde != '':
            cursor.execute("INSERT INTO Temperatuur2arduino1 (Temperatuur2arduino1, date) VALUES (%s, %s)",
                           (Temperatuur2arduino1_waarde, timestamp))
            conn.commit()
            print("Temperatuur waarde is ", Temperatuur2arduino1_waarde)
    except:
        print('Temp error')
    try:
        if LuchtVocht2arduino1_waarde != '':
            cursor.execute("INSERT INTO LuchtVocht2arduino1 (LuchtVocht2arduino1, date) VALUES (%s, %s)",
                           (LuchtVocht2arduino1_waarde, timestamp))
            conn.commit()
            print("LuchtVocht is ", LuchtVocht2arduino1_waarde)
    except:
        print('LV error')
    try:
        if Licht2arduino1_waarde != '':
            cursor.execute("INSERT INTO Licht2arduino1 (Licht2arduino1, date) VALUES (%s, %s)",
                           (Licht2arduino1_waarde, timestamp))
            conn.commit()
            print("Licht waarde is ", Licht2arduino1_waarde)
    except:
        print('Licht error')


def import_arduino2():
    try:
        conn = mysql.connector.connect(db)  # handig voor later in de code
        if AardeVocht1arduino2_waarde != '':
            cursor.execute("INSERT INTO AardeVocht1arduino2 (AardeVocht1arduino2, date) VALUES (%s, %s)",
                           (AardeVocht1arduino2_waarde, timestamp))
            conn.commit()
            print("Aardevocht waarde is ", AardeVocht1arduino2_waarde)

    except:
        print('AardeVocht error')

    try:
        conn = mysql.connector.connect(db)
        if Temperatuur1arduino2_waarde != '':
            cursor.execute("INSERT INTO Temperatuur1arduino2 (Temperatuur1arduino2, date) VALUES (%s, %s)",
                           (Temperatuur1arduino2_waarde, timestamp))
            conn.commit()
            print("Temperatuur waarde is ", Temperatuur1arduino2_waarde)
    except:
        print('Temp error')

    try:
        conn = mysql.connector.connect(db)
        if LuchtVocht1arduino2_waarde != '':
            cursor.execute("INSERT INTO LuchtVocht1arduino2 (LuchtVocht1arduino2, date) VALUES (%s, %s)",
                           (LuchtVocht1arduino2_waarde, timestamp))
            conn.commit()
            print("LuchtVocht is ", LuchtVocht1arduino2_waarde)
    except:
        print('LV error')
    try:
        conn = mysql.connector.connect(db)
        if Licht1arduino2_waarde != '':
            cursor.execute("INSERT INTO Licht1arduino2 (Licht1arduino2, date) VALUES (%s, %s)",
                           (Licht1arduino2_waarde, timestamp))
            conn.commit()
            print("Licht waarde is ", Licht1arduino2_waarde)
    except:
        print('Licht error')
    try:
        conn = mysql.connector.connect(db)
        if AardeVocht2arduino2_waarde != '':
            cursor.execute("INSERT INTO AardeVocht2arduino2 (AardeVocht2arduino2, date) VALUES (%s, %s)",
                           (AardeVocht2arduino2_waarde, timestamp))
            conn.commit()
            print("Aardevocht waarde is ", AardeVocht2arduino2_waarde)

    except:
        print('AardeVocht error')

    try:
        conn = mysql.connector.connect(db)
        if Temperatuur2arduino2_waarde != '':
            cursor.execute("INSERT INTO Temperatuur2arduino2 (Temperatuur2arduino2, date) VALUES (%s, %s)",
                           (Temperatuur2arduino2_waarde, timestamp))
            conn.commit()
            print("Temperatuur waarde is ", Temperatuur2arduino2_waarde)
    except:
        print('Temp error')
    try:
        conn = mysql.connector.connect(db)
        if LuchtVocht2arduino2_waarde != '':
            cursor.execute("INSERT INTO LuchtVocht2arduino2 (LuchtVocht2arduino2, date) VALUES (%s, %s)",
                           (LuchtVocht2arduino2_waarde, timestamp))
            conn.commit()
            print("LuchtVocht is ", LuchtVocht2arduino2_waarde)
    except:
        print('LV error')
        conn = mysql.connector.connect(db)
    try:
        if Licht2arduino2_waarde != '':
            cursor.execute("INSERT INTO Licht2arduino2 (Licht2arduino2, date) VALUES (%s, %s)",
                           (Licht2arduino2_waarde, timestamp))
            conn.commit()
            print("Licht waarde is ", Licht2arduino2_waarde)
    except:
        print('Licht error')


def connect_arduinos():
    # arduino inloggen
    # Arduino(s) connecten
    arduino1 = serial.Serial("COM5", baudrate=9600, timeout=3)
    arduino2 = serial.Serial("COM3", baudrate=9600, timeout=3)  # idk wat timeout doet
    arduino3 = serial.Serial("COM4", baudrate=9600, timeout=3)  # deze klopt niet

    arduinoDataString1 = arduino1.readline().decode('utf-8').strip()  # vertaald de arduino wartaal
    arduinoDataString2 = arduino2.readline().decode('utf-8').strip()  # vertaald de arduino wartaal
    arduinoDataString3 = arduino3.readline().decode('utf-8').strip()  # vertaald de arduino wartaal


# Begrippen definieren om eventuele foutmeldingen te voorkomen
arduinoDataString1 = 0
arduinoDataString2 = 0
arduinoDataString3 = 0

if x == 1:  # dit voorkomt gedoe met tab
    try:
        arduino1 = serial.Serial("COM5", baudrate=9600, timeout=3)
        arduino1.write(
            ("Initiate sensorsimport ArduinoAS").encode('utf-8'))  # moet ik nog veranderen bij de arduino code
    except:
        if language == 'English':
            st.write("Arduino1 not available, please check connection")
        if language == 'Nederlands':
            st.write("Arduino1 is niet beschikbaar, controleer de verbinding")
    try:
        arduino2 = serial.Serial("COM3", baudrate=9600, timeout=3)  # idk wat timeout doet
        arduino2.write(("Initiate sensorsimport Arduino 1").encode('utf-8'))
    except:
        if language == 'English':
            st.write("Arduino2 not available, please check connection")
        if language == 'Nederlands':
            st.write("Arduino2 is niet beschikbaar, controleer de verbinding")
    try:
        arduino3 = serial.Serial("COM4", baudrate=9600, timeout=3)  # deze klopt niet
        arduino3.write(("Initiate sensorsimport Arduino 2").encode('utf-8'))
    except:
        if language == 'English':
            st.write("Arduino3 not available, please check connection")
        if language == 'Nederlands':
            st.write("Arduino3 is niet beschikbaar, controleer de verbinding")

    if sideselectbox == "Home":
        # App openen
        try:
            if language == 'English':
                st.title("The GrowGo Response Unit")
                st.header("Our new way to emergency aid for a world with less hunger")
                st.write('Welcome to GrowGo, home of the innovative GrowGo Response Unit (GRU). Designed to provide fast, reliable access to emergency food supplies, the GRU is a portable, self-sustaining unit that allows you to grow crops quickly in any environment. Whether facing natural disasters, food shortages, or challenging climates, our GRU is engineered to ensure fresh, nutritious food is always within reach. With easy setup and rapid growth capabilities, GrowGo is your trusted partner in food security, delivering resilience when you need it most.')
                image = Image.open('images/AI growgo.jpg')
                st.image(image, caption="AI Generated" ,use_column_width=True)

            if language == 'Nederlands':
                st.title("De GrowGo Response Unit")
                st.header("Onze nieuwe weg om noodhulp te bieden voor een wereld met minder honger")
                st.write('Welkom bij GrowGo, de thuisbasis van de innovatieve GrowGo Response Unit (GRU). De GRU is ontworpen om snel en betrouwbaar toegang te bieden tot noodvoedselvoorraden. Het is een draagbare, zelfvoorzienende unit waarmee je snel gewassen kunt kweken in elke omgeving. Of je nu te maken hebt met natuurrampen, voedseltekorten of moeilijke klimaten, onze GRU is ontworpen om ervoor te zorgen dat vers, voedzaam voedsel altijd binnen handbereik is. Dankzij de eenvoudige installatie en snelle groeimogelijkheden is GrowGo je betrouwbare partner in voedselzekerheid, en biedt het weerstand wanneer je het het meest nodig hebt.')
                image = Image.open('images/AI growgo.jpg')
                st.image(image, caption="AI Gegenereerd" ,use_column_width=True)
        except:
            if language == 'English':
                print("Error 101, contact website builders")
            if language == 'Nederlands':
                print("Error 101, neem contact op met de website ontwikkelaars")
    # Begin import
    elif sideselectbox == "Realtime plot and input":
        # Ontcijferen van input, allemaal in integers anders foutmeldingen
        try:
            while x == 1:
                connect_arduinos()
                try:
                    # Dataimport arduino1 AKA arduino AS
                    print(arduinoDataString1)
                    AardeVochtASwaarde = arduinoDataString1[0:3]
                    TemperatuurASwaarde = arduinoDataString1[3:5]
                    LuchtVochtASwaarde = arduinoDataString1[8:10]
                    LichtASwaarde = arduinoDataString1[13:16]
                    import_arduinoAS()
                    # Regendruppelaar1 = arduinoDataString[16:20]
                except:
                    st.write("Error data waardes 1")

                try:
                    AardeVocht1arduino1_waarde = arduinoDataString2[
                                                 0:3]  # drie waardes want 1000 is onwaarschijnlijk tenzij kapot
                    Licht1arduino1_waarde = arduinoDataString2[3:6]
                    LuchtVocht1arduino1_waarde = arduinoDataString2[6:8]
                    Temperatuur1arduino1_waarde = arduinoDataString2[8:10]  # klein gat door .00
                    AardeVocht2arduino1_waarde = (arduinoDataString2[13:15])
                    Licht2arduino1_waarde = arduinoDataString2[15:18]
                    LuchtVocht2arduino1_waarde = arduinoDataString2[18:20]
                    Temperatuur2arduino1_waarde = arduinoDataString2[23:25]  # klein gat door .00
                    import_arduino1()
                except:
                    st.write("Error data waardes 2")

                # Hier begint de arduino 3

                try:
                    AardeVocht1arduino2_waarde = arduinoDataString3[
                                                 0:3]  # drie waardes want 1000 is onwaarschijnlijk tenzij kapot
                    Licht1arduino2_waarde = arduinoDataString3[3:6]
                    LuchtVocht1arduino2_waarde = arduinoDataString3[6:8]
                    Temperatuur1arduino2_waarde = arduinoDataString3[8:10]  # klein gat door .00
                    AardeVocht2arduino2_waarde = (arduinoDataString3[13:15])
                    Licht2arduino2_waarde = arduinoDataString3[15:18]
                    LuchtVocht2arduino2_waarde = arduinoDataString3[18:20]
                    Temperatuur2arduino2_waarde = arduinoDataString3[23:25]  # klein gat door .00
                    import_arduino2()
                except:
                    st.write("Error data waardes 3")
        except:
            st.write("Error 101, contact website builders")

        # Nu de dataimport is gestart, moeten we de realtime data plot gaan plotten. Hiervoor heb je een lijst nodig:
        for i in range(1,
                       100):  # while len(TemperatuurASList) >= 25 or len(TemperatuurArduino1Waarde1List) >= 25 or len(TemperatuurArduino2Waarde1List):
            try:
                TemperatuurASList = TemperatuurASList.append(TemperatuurASwaarde)
                AardeVochtASList = AardeVochtASList.append(AardeVochtASwaarde)
                LuchtVochtASList = LuchtVochtASList.append(LuchtVochtASwaarde)
                LichtASList = LichtASList.append(LichtASwaarde)

                TemperatuurASdate = TemperatuurASdate.append(timestamp)
                AardeVochtASdate = AardeVochtASdate.append(timestamp)
                LuchtVochtASdate = LuchtVochtASdate.append(timestamp)
                LichtASdate = LichtASdate.append(timestamp)

                TemperatuurArduino1Waarde1List = TemperatuurArduino1Waarde1List.apppend(Temperatuur1arduino1_waarde)
                AardeVochtArduino1Waarde1List = AardeVochtArduino1Waarde1List.append(AardeVocht1arduino1_waarde)
                LuchtVochtArduino1Waarde1List = LuchtVochtArduino1Waarde1List.append(LuchtVocht1arduino1_waarde)
                LichtArduino1Waarde1List = LichtArduino1Waarde1List.append(Licht1arduino1_waarde)

                TemperatuurArduino1Waarde1Date = TemperatuurArduino1Waarde1Date.append(timestamp)
                AardeVochtArduino1Waarde1Date = AardeVochtArduino1Waarde1Date.append(timestamp)
                LuchtVochtArduino1Waarde1Date = LuchtVochtArduino1Waarde1Date.append(timestamp)
                LichtArduino1Waarde1Date = LichtArduino1Waarde1Date.append(timestamp)

                TemperatuurArduino1Waarde2List = TemperatuurArduino1Waarde2List.apppend(Temperatuur2arduino2_waarde)
                AardeVochtArduino1Waarde2List = AardeVochtArduino1Waarde2List.append(AardeVocht2arduino2_waarde)
                LuchtVochtArduino1Waarde2List = LuchtVochtArduino1Waarde2List.append(LuchtVocht2arduino2_waarde)
                LichtArduino1Waarde2List = LichtArduino1Waarde2List.append(Licht2arduino2_waarde)

                TemperatuurArduino1Waarde2Date = TemperatuurArduino1Waarde2Date.append(timestamp)
                AardeVochtArduino1Waarde2Date = AardeVochtArduino1Waarde2Date.append(timestamp)
                LuchtVochtArduino1Waarde2Date = LuchtVochtArduino1Waarde2Date.append(timestamp)
                LichtArduino1Waarde2Date = LichtArduino1Waarde2Date.append(timestamp)

                TemperatuurArduino2Waarde1List = TemperatuurArduino2Waarde1List.apppend(Temperatuur1arduino2_waarde)
                AardeVochtArduino2Waarde1List = AardeVochtArduino2Waarde1List.append(AardeVocht1arduino2_waarde)
                LuchtVochtArduino2Waarde1List = LuchtVochtArduino2Waarde1List.append(LuchtVocht1arduino2_waarde)
                LichtArduino2Waarde1List = LichtArduino2Waarde1List.append(Licht1arduino2_waarde)

                TemperatuurArduino2Waarde1Date = TemperatuurArduino2Waarde1Date.append(timestamp)
                AardeVochtArduino2Waarde1Date = AardeVochtArduino2Waarde1Date.append(timestamp)
                LuchtVochtArduino2Waarde1Date = LuchtVochtArduino2Waarde1Date.append(timestamp)
                LichtArduino2Waarde2Date = LichtArduino2Waarde1Date.append(timestamp)

                TemperatuurArduino2Waarde2List = TemperatuurArduino2Waarde2List.apppend(Temperatuur2arduino2_waarde)
                AardeVochtArduino2Waarde2List = AardeVochtArduino2Waarde2List.append(AardeVocht2arduino2_waarde)
                LuchtVochtArduino2Waarde2List = LuchtVochtArduino2Waarde2List.append(LuchtVocht2arduino2_waarde)
                LichtArduino2Waarde2List = LichtArduino2Waarde2List.append(Licht2arduino2_waarde)

                TemperatuurArduino2Waarde2Date = TemperatuurArduino2Waarde2Date.append(timestamp)
                AardeVochtArduino2Waarde2Date = AardeVochtArduino2Waarde2Date.append(timestamp)
                LuchtVochtArduino2Waarde2Date = LuchtVochtArduino2Waarde2Date.append(timestamp)
                LichtArduino2Waarde2Date = LichtArduino2Waarde2Date.append(timestamp)

            except:
                st.write("List appending for Realtime plotting failed")
            # We beginnen nu met het plotten
        # try:
        #    fig,ax = plt.subplots(4, 5)
        #   ax[1,0].plot(TemperatuurASdate, TemperatuurASList)
        #   ax[1,1].plot(AardeVochtASdate,)
        # except:
        # st.write("Failedprinting realtime plots")




    elif sideselectbox == "Plot Data":
        try:
            # Knopje om csv of txt van SD kaart in SQL te stoppen
            # Er mogen geen andere gekke tekens bij zijn die niet voorkomen in de arduino output
            FileButton = st.file_uploader(":file_folder Upload SD-card in csv format",
                                          type=(["csv", "txt", "xlsx", "xls"]))
            SDimport = st.selectbox("Which arduino would you like to import?",
                                    ['ArduinoAS', 'Arduino2', 'Arduino3', 'ArduinoAS.txt', 'Arduino2.txt',
                                     'Arduino3.txt'])
            if FileButton is not None:
                filename = FileButton.name
                st.write(filename)
                NewFile = pd.read_csv(filename, encoding="ISO-8859-1")
                if SDimport == 'ArduinoAS':
                    arduino1 = serial.Serial("COM5", baudrate=9600, timeout=3)
                    Pycharmcommand = 'Initiate SD-card ArduinoAS'
                    arduino1.write(Pycharmcommand.encode())
                    arduinoDataString1 = arduino1.readline().decode('utf-8').strip()  # vertaald de arduino wartaal
                    for i in range(arduinoDataString1):
                        import_arduinoAS()
                elif SDimport == 'Arduino2':
                    arduino2 = serial.Serial("COM4", baudrate=9600, timeout=3)
                    Pycharmcommand = 'Initiate SD-card Arduino 2'
                    arduino2.write(Pycharmcommand.encode())
                    arduinoDataString2 = arduino2.readline().decode('utf-8').strip()  # vertaald de arduino wartaal
                    for i in range(arduinoDataString2):
                        import_arduino1()
                elif SDimport == 'Arduino3':
                    arduino3 = serial.Serial("COM3", baudrate=9600, timeout=3)
                    Pycharmcommand = 'Initiate SD-card Arduino 3'
                    arduino3.write(Pycharmcommand.encode())
                    arduinoDataString3 = arduino3.readline().decode('utf-8').strip()  # vertaald de arduino wartaal
                    for i in range(arduinoDataString3):
                        import_arduino2()
                elif SDimport == 'Arduino2.txt':
                    with open(filename) as file:
                        for row in file:
                            print('zero')
                            AardeVocht1arduino1_waarde = file[
                                                         0:3]  # drie waardes want 1000 is onwaarschijnlijk tenzij kapot
                            Licht1arduino1_waarde = file[3:6]
                            LuchtVocht1arduino1_waarde = file[6:8]
                            Temperatuur1arduino1_waarde = file[8:10]  # klein gat door .00
                            AardeVocht2arduino1_waarde = file[13:15]
                            Licht2arduino1_waarde = file[15:18]
                            LuchtVocht2arduino1_waarde = file[18:20]
                            Temperatuur2arduino1_waarde = arduinoDataString2[23:25]  # klein gat door .00
                            import_arduino1()
                elif SDimport == 'Arduino3.txt':
                    with open(filename) as file:
                        AardeVocht1arduino2_waarde = file[
                                                     0:3]  # drie waardes want 1000 is onwaarschijnlijk tenzij kapot
                        Licht1arduino2_waarde = file[3:6]
                        LuchtVocht1arduino2_waarde = file[6:8]
                        Temperatuur1arduino2_waarde = file[8:10]  # klein gat door .00
                        AardeVocht2arduino2_waarde = file[13:15]
                        Licht2arduino2_waarde = file[15:18]
                        LuchtVocht2arduino2_waarde = file[18:20]
                        Temperatuur2arduino2_waarde = arduinoDataString2[23:25]  # klein gat door .00
                        import_arduino2()
                elif SDimport == 'ArduinoAS.txt':
                    with open(filename) as file:
                        AardeVocht1ASwaarde = file[0:3]  # drie waardes want 1000 is onwaarschijnlijk tenzij kapot
                        Licht1ASwaarde = file[3:6]
                        LuchtVocht1ASwaarde = file[6:8]
                        Temperatuur1ASwaarde = file[8:10]  # klein gat door .00
                        AardeVocht2ASwaarde = file[13:15]
                        Licht2ASwaarde = file[15:18]
                        LuchtVocht2ASwaarde = file[18:20]
                        Temperatuur2ASwaarde = arduinoDataString2[23:25]
                        import_arduinoAS()
            else:
                st.write("No new file upload")
            # De plotchaos
            cursor.execute("SELECT * FROM TemperatuurAS, ORDER BY date DESC")
            rows_TemperatuurAS = cursor.fetchall()
            df_TemperatuurAS = pd.DataFrame(rows_TemperatuurAS)
            df_TemperatuurAS.columns = ['date', 'TemperatuurAS_waarde']

            cursor.execute("SELECT * FROM AardeVochtAS, ORDER BY date DESC")
            rows_AardeVochtAS = cursor.fetchall()
            df_AardeVochtAS = pd.DataFrame(rows_AardeVochtAS)
            df_AardeVochtAS.columns = ['date', 'AardeVochtAS_waarde']

            cursor.execute("SELECT * FROM LuchtVochtAS, ORDER BY date DESC")
            rows_LuchtVochtAS = cursor.fetchall()
            df_LuchtVochtAS = pd.DataFrame(rows_LuchtVochtAS)
            df_LuchtVochtAS.columns = ['date', 'LuchtVochtAS_waarde']

            cursor.execute("SELECT * FROM LichtAS, ORDER BY date DESC")
            rows_LichtAS = cursor.fetchall()
            df_LichtAS = pd.DataFrame(rows_LichtAS)
            df_LichtAS.columns = ['date', 'LichtAS_waarde']

            cursor.execute("SELECT * FROM Temperatuur1Arduino1_waarde, ORDER BY date DESC")
            rows_Temperatuur1Arduino1_waarde = cursor.fetchall()
            df_Temperatuur1Arduino1_waarde = pd.DataFrame(rows_Temperatuur1Arduino1_waarde)
            df_Temperatuur1Arduino1_waarde.columns = ['date', 'Temperatuur1Arduino1_waarde']

            cursor.execute("SELECT * FROM Licht1Arduino1_waarde, ORDER BY date DESC")
            rows_Licht1Arduino1_waarde = cursor.fetchall()
            df_Licht1Arduino1_waarde = pd.DataFrame(rows_Licht1Arduino1_waarde)
            df_Licht1Arduino1_waarde.columns = ['date', 'Licht1Arduino1_waarde']

            cursor.execute("SELECT * FROM AardeVocht1Arduino1_waarde, ORDER BY date DESC")
            rows_AardeVocht1Arduino1_waarde = cursor.fetchall()
            df_AardeVocht1Arduino1_waarde = pd.DataFrame(rows_AardeVocht1Arduino1_waarde)
            df_AardeVocht1Arduino1_waarde.columns = ['date', 'AardeVocht1Arduino1_waarde']

            cursor.execute("SELECT * FROM LuchtVocht1Arduino1_waarde, ORDER BY date DESC")
            rows_LuchtVocht1Arduino1_waarde = cursor.fetchall()
            df_LuchtVocht1Arduino1_waarde = pd.DataFrame(rows_LuchtVocht1Arduino1_waarde)
            df_LuchtVocht1Arduino1_waarde.columns = ['date', 'LuchtVocht1Arduino1_waarde']

            cursor.execute("SELECT * FROM Temperatuur2Arduino1_waarde, ORDER BY date DESC")
            rows_Temperatuur2Arduino1_waarde = cursor.fetchall()
            df_Temperatuur2Arduino1_waarde = pd.DataFrame(rows_Temperatuur2Arduino1_waarde)
            df_Temperatuur2Arduino1_waarde.columns = ['date', 'Temperatuur2Arduino1_waarde']

            cursor.execute("SELECT * FROM Licht2Arduino1_waarde, ORDER BY date DESC")
            rows_Licht2Arduino1_waarde = cursor.fetchall()
            df_Licht2Arduino1_waarde = pd.DataFrame(rows_Licht2Arduino1_waarde)
            df_Licht2Arduino1_waarde.columns = ['date', 'Licht2Arduino1_waarde']

            cursor.execute("SELECT * FROM AardeVocht2Arduino1_waarde, ORDER BY date DESC")
            rows_AardeVocht2Arduino1_waarde = cursor.fetchall()
            df_AardeVocht2Arduino1_waarde = pd.DataFrame(rows_AardeVocht2Arduino1_waarde)
            df_AardeVocht2Arduino1_waarde.columns = ['date', 'AardeVocht2Arduino1_waarde']

            cursor.execute("SELECT * FROM LuchtVocht2Arduino1_waarde, ORDER BY date DESC")
            rows_LuchtVocht2Arduino1_waarde = cursor.fetchall()
            df_LuchtVocht2Arduino1_waarde = pd.DataFrame(rows_LuchtVocht2Arduino1_waarde)
            df_LuchtVocht2Arduino1_waarde.columns = ['date', 'LuchtVocht2Arduino1_waarde']

            cursor.execute("SELECT * FROM Temperatuur1Arduino2_waarde, ORDER BY date DESC")
            rows_Temperatuur1Arduino2_waarde = cursor.fetchall()
            df_Temperatuur1Arduino2_waarde = pd.DataFrame(rows_Temperatuur1Arduino2_waarde)
            df_Temperatuur1Arduino2_waarde.columns = ['date', 'Temperatuur1Arduino2_waarde']

            cursor.execute("SELECT * FROM Licht1Arduino2_waarde, ORDER BY date DESC")
            rows_Licht1Arduino2_waarde = cursor.fetchall()
            df_Licht1Arduino2_waarde = pd.DataFrame(rows_Licht1Arduino2_waarde)
            df_Licht1Arduino2_waarde.columns = ['date', 'Licht1Arduino2_waarde']

            cursor.execute("SELECT * FROM AardeVocht1Arduino2_waarde, ORDER BY date DESC")
            rows_AardeVocht1Arduino2_waarde = cursor.fetchall()
            df_AardeVocht1Arduino2_waarde = pd.DataFrame(rows_AardeVocht1Arduino2_waarde)
            df_AardeVocht1Arduino2_waarde.columns = ['date', 'AardeVocht1Arduino2_waarde']

            cursor.execute("SELECT * FROM LuchtVocht1Arduino2_waarde, ORDER BY date DESC")
            rows_LuchtVocht1Arduino2_waarde = cursor.fetchall()
            df_LuchtVocht1Arduino2_waarde = pd.DataFrame(rows_LuchtVocht1Arduino2_waarde)
            df_LuchtVocht1Arduino2_waarde.columns = ['date', 'LuchtVocht1Arduino2_waarde']

            cursor.execute("SELECT * FROM Temperatuur2Arduino2_waarde, ORDER BY date DESC")
            rows_Temperatuur2Arduino2_waarde = cursor.fetchall()
            df_Temperatuur2Arduino2_waarde = pd.DataFrame(rows_Temperatuur2Arduino2_waarde)
            df_Temperatuur2Arduino2_waarde.columns = ['date', 'Temperatuur2Arduino2_waarde']

            cursor.execute("SELECT * FROM Licht2Arduino2_waarde, ORDER BY date DESC")
            rows_Licht2Arduino2_waarde = cursor.fetchall()
            df_Licht2Arduino2_waarde = pd.DataFrame(rows_Licht2Arduino2_waarde)
            df_Licht2Arduino2_waarde.columns = ['date', 'Licht2Arduino2_waarde']

            cursor.execute("SELECT * FROM AardeVocht2Arduino2_waarde, ORDER BY date DESC")
            rows_AardeVocht2Arduino2_waarde = cursor.fetchall()
            df_AardeVocht2Arduino2_waarde = pd.DataFrame(rows_AardeVocht2Arduino2_waarde)
            df_AardeVocht2Arduino2_waarde.columns = ['date', 'AardeVocht2Arduino2_waarde']

            cursor.execute("SELECT * FROM LuchtVocht2Arduino2_waarde, ORDER BY date DESC")
            rows_LuchtVocht2Arduino2_waarde = cursor.fetchall()
            df_LuchtVocht2Arduino2_waarde = pd.DataFrame(rows_LuchtVocht2Arduino2_waarde)
            df_LuchtVocht2Arduino2_waarde.columns = ['date', 'LuchtVocht2Arduino2_waarde']

            fig, ax = plt.subplots(5, 4)
            ax[1, 0].plot(df_TemperatuurAS['TemperatuurAS_waarde'], df_TemperatuurAS('date'), color='r')
            ax[1, 1].plot(df_AardeVochtAS['AardeVochtAS_waarde'], df_AardeVochtAS('date'), color='b')
            ax[1, 2].plot(df_LuchtVochtAS['LuchtVochtAS_waarde'], df_LuchtVochtAS('date'), color='b')
            ax[1, 3].plot(df_LichtAS['LichtAS_waarde'], df_LichtAS('date'), color='r')

            ax[2, 0].plot(df_Temperatuur1Arduino1_waarde['Temperatuur1Arduino1_waarde'],
                          df_Temperatuur1Arduino1_waarde('date'), color='r')
            ax[2, 1].plot(df_AardeVocht1Arduino1_waarde['AardeVocht1Arduino1_waarde'],
                          df_AardeVocht1Arduino1_waarde('date'), color='b')
            ax[2, 2].plot(df_LuchtVocht1Arduino1_waarde['LuchtVocht1Arduino1_waarde'],
                          df_LuchtVocht1Arduino1_waarde('date'), color='b')
            ax[2, 3].plot(df_Licht1Arduino1_waarde['Licht1Arduino1_waarde'], df_Licht1Arduino1_waarde('date'),
                          color='r')

            ax[3, 0].plot(df_Temperatuur2Arduino1_waarde['Temperatuur2Arduino1_waarde'],
                          df_Temperatuur2Arduino1_waarde('date'), color='r')
            ax[3, 1].plot(df_AardeVocht2Arduino1_waarde['AardeVocht2Arduino1_waarde'],
                          df_AardeVocht2Arduino1_waarde('date'), color='b')
            ax[3, 2].plot(df_LuchtVocht2Arduino1_waarde['LuchtVocht2Arduino1_waarde'],
                          df_LuchtVocht2Arduino1_waarde('date'), color='b')
            ax[3, 3].plot(df_Licht1Arduino1_waarde['Licht2Arduino1_waarde'], df_Licht2Arduino1_waarde('date'),
                          color='r')

            ax[4, 0].plot(df_Temperatuur1Arduino2_waarde['Temperatuur1Arduino2_waarde'],
                          df_Temperatuur1Arduino2_waarde('date'),
                          color='r')
            ax[4, 1].plot(df_AardeVocht1Arduino2_waarde['AardeVocht1Arduino2_waarde'],
                          df_AardeVocht1Arduino2_waarde('date'),
                          color='b')
            ax[4, 2].plot(df_LuchtVocht1Arduino2_waarde['LuchtVocht1Arduino2_waarde'],
                          df_LuchtVocht1Arduino2_waarde('date'),
                          color='b')
            ax[4, 3].plot(df_Licht1Arduino2_waarde['Licht1Arduino2_waarde'], df_Licht1Arduino2_waarde('date'),
                          color='r')

            ax[5, 0].plot(df_Temperatuur2Arduino2_waarde['Temperatuur2Arduino2_waarde'],
                          df_Temperatuur2Arduino2_waarde('date'), color='r')
            ax[5, 1].plot(df_AardeVocht2Arduino2_waarde['AardeVocht2Arduino2_waarde'],
                          df_AardeVocht2Arduino2_waarde('date'), color='b')
            ax[5, 2].plot(df_LuchtVocht2Arduino2_waarde['LuchtVocht2Arduino2_waarde'],
                          df_LuchtVocht2Arduino2_waarde('date'), color='b')
            ax[5, 3].plot(df_Licht1Arduino2_waarde['Licht2Arduino2_waarde'], df_Licht2Arduino2_waarde('date'),
                          color='r')

            st.pyplot(fig)
        except:
            st.write("Error 101, contact website builders")

    elif sideselectbox == "Progression and prospects":

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
