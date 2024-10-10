import mysql.connector

db = mysql.connector.connect(  host='localhost',
  user='root',
  password="theGrowGo2024Casper",
    database = "growgo1",
    auth_plugin = 'mysql_native_password' )

cursor = db.cursor()

cursor.execute('CREATE TABLE AardeVochtAS (AardeVochtASwaarde smallint, date Bigint UNSIGNED )')
cursor.execute('CREATE TABLE TemperatuurAS (TemperatuurASwaarde smallint, date Bigint UNSIGNED )')
cursor.execute('CREATE TABLE LuchtVochtAS (LuchtVochtASwaarde smallint, date Bigint UNSIGNED )')
cursor.execute('CREATE TABLE LichtAS (LichtASwaarde smallint, date Bigint)')
cursor.execute('CREATE TABLE Temperatuur1Arduino1(Temperatuur1Arduino1_waarde smallint, date Bigint UNSIGNED )')
cursor.execute('CREATE TABLE Temperatuur2Arduino1(Temperatuur2Arduino1_waarde smallint, date Bigint UNSIGNED )')
cursor.execute('CREATE TABLE Temperatuur1Arduino2(Temperatuur1Arduino2_waarde smallint, date Bigint UNSIGNED )')
cursor.execute('CREATE TABLE Temperatuur2Arduino2(Temperatuur2Arduino2_waarde smallint, date Bigint UNSIGNED )')
cursor.execute('CREATE TABLE LuchtVocht1Arduino1(LuchtVocht1Arduino1_waarde smallint, date Bigint UNSIGNED )')
cursor.execute('CREATE TABLE LuchtVocht2Arduino1(LuchtVocht2Arduino1_waarde smallint, date Bigint UNSIGNED )')
cursor.execute('CREATE TABLE LuchtVocht2Arduino2(LuchtVocht2Arduino2_waarde smallint, date Bigint UNSIGNED )')
cursor.execute('CREATE TABLE LuchtVocht1Arduino2(LuchtVocht1Arduino2_waarde smallint, date Bigint UNSIGNED )')
cursor.execute('CREATE TABLE Licht1Arduino1(Licht1Arduino1_waarde smallint, date Bigint UNSIGNED )')
cursor.execute('CREATE TABLE Licht2Arduino1(Licht2Arduino1_warden smallint, date Bigint UNSIGNED )')
cursor.execute('CREATE TABLE Licht1Arduino2(Licht1Arduino2_waarde smallint, date Bigint UNSIGNED )')
cursor.execute('CREATE TABLE Licht2Arduino2(Licht2Arduino2_waarde smallint, date Bigint UNSIGNED )')
cursor.execute('CREATE TABLE AardeVocht1Arduino1(AardeVocht1Arduino1_waarde smallint, date Bigint UNSIGNED )')
cursor.execute('CREATE TABLE AardeVocht2Arduino1(AardeVocht2Arduino1_waarde smallint, date Bigint UNSIGNED )')
cursor.execute('CREATE TABLE AardeVocht2Arduino2(AardeVocht2Arduino2_waarde smallint, date Bigint UNSIGNED )')
cursor.execute('CREATE TABLE AardeVocht1Arduino2(AardeVocht1Arduino2_waarde smallint, date Bigint UNSIGNED )')
db.commit()