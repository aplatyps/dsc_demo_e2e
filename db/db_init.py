import pandas
import dbconnect as dbc

conn = dbc.create_connection()

csvfile_path = 'emscad_v1.csv'
df_all = pandas.read_csv(csvfile_path)
df_all.to_sql('emscad_all', conn, if_exists='replace', index=False)

dbc.view_db(conn)

dbc.close_connection(conn)