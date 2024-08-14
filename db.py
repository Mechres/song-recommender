import sqlite3
import pandas as pd
import os

def create_database():
    db_path = os.path.join(os.getcwd(), 'songs_database.db')
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS songs (
                id TEXT PRIMARY KEY, 
                name TEXT, 
                artist TEXT, 
                popularity REAL, 
                genres TEXT,
                danceability REAL, 
                energy REAL, 
                key REAL, 
                loudness REAL, 
                mode REAL, 
                speechiness REAL, 
                acousticness REAL, 
                instrumentalness REAL, 
                liveness REAL, 
                valence REAL, 
                tempo REAL
            )
        ''')
        conn.commit()

        xlsx_file = 'songs_dataset.xlsx'  # Updated file name
        if os.path.exists(xlsx_file):
            print("XLSX file found, proceeding to load data...")
            df = pd.read_excel(xlsx_file)  # Read excel instead of csv

            # Drop duplicates (Choose the most suitable method as explained earlier)
            df.drop_duplicates(subset='id', keep='first', inplace=True)

            # Check if the table is empty before inserting data
            if not cursor.execute("SELECT * FROM songs").fetchone():
                df.to_sql('songs', conn, if_exists='append', index=False)
                conn.commit()
                print("Data successfully loaded into the database.")
            else:
                print("Table 'songs' already contains data. Skipping data insertion.")
        else:
            print(f"XLSX file not found: {xlsx_file}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()