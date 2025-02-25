import os
import pandas as pd
from pathlib import Path

# Define input file pattern (adjust path if needed)
data_dir = Path('data')
cleaned_data_dir = Path('cleaned_data')
input_files = list(data_dir.glob("Stats La FOTTA EUCF 2024 - *.csv"))
tot = "data/Stats La FOTTA EUCF 2024 - Tot.csv"

# Initialize data structures
# players = []
# stats =[]
# games = []
# points = []
# possessions = []
# events = []

# Players
players = pd.read_csv(tot, usecols=[0,1])
players.rename(columns={'Unnamed: 0': 'jersey_number', 'Unnamed: 1': 'name'}, inplace=True)
players['team'] = 'BFD La Fotta'
players.dropna(inplace=True)
players['jersey_number'] = players['jersey_number'].astype('int')
players = players[['name', 'jersey_number','team']]
# print(players.to_string())

players_df = pd.DataFrame(players)
players_df.to_csv(cleaned_data_dir / "players.csv", index=False)
# players_df.to_csv("C:\\Users\\Alessandro\\Documents\\ale\\valuationLaFotta2024\\cleaned_data\\players.csv", index=False)




# def_stats.csv
# Game Stats Table (stats.csv)
# id
# name
# short_name (A, M, T, etc.)
# owner (Is it a player or the team?)
# recurrency (Game-level, point-level, possession-level?) : (gm, pt, ps)  =>  can be array for multiple rec? NOPE
stat_names = []
stat_short_names = []
stat_owners = []
stat_recurrency = []
# Taken Manually from pd.read_csv("C:\\Users\\Alessandro\\Documents\\ale\\peLaFottaEUCF2024\\Stats La FOTTA EUCF 2024\\Stats La FOTTA EUCF 2024 - BADSKID.csv")
# player stats 20 elems
stat_short_names = ['S', 'D', 'HD', 'LT', 'T', 'P', 'Assist', 'Mete', 'Mete g att','Mete g dif','Mete g tot',"Possesses Played Offence","Possesses Played Defence","Possesses Played Tot", 'Mga att (min)', '(sec)', 'Mga dif (min)', '(sec).1', 'Mga tot (min)', '(sec).2']
stat_names = ["super","block","help_defence","terrible_throw","throwaway","drop","assist","score","pts_played_o","pts_played_d","pts_played_tot","pss_played_o","pss_played_d","pss_played_tot","min_played_o","sec_played_o","min_played_d","sec_played_d","min_played_tot","sec_played_tot"]
stat_recurrency = ["pt","pt","pt","pt","pt","pt","ps","ps","gm","gm","gm","pt","pt","pt","pt","pt","pt","pt","pt","pt"] # ps=ps+pt+gm, pt=pt+gm, gm=gm
stat_dtype = ["int"]*len(stat_names)
stat_owners = ["player"]*len(stat_names)
stat_important = ['', '!', '!', '!', '!', '!', '', '', '', '', '!', '', '', '!', '', '', '', '', '', ''] # [""]*len(stat_names)
#team stats: 19 elems, Lineup.dtype = [], Break.dtype = bool
t_stat_short_names = ["In Campo", "Blocks","Tovs","Break","A/D","","","","","","Schema A","Schema D","Storia neg","Storia pos","Assist","Meta","","","","","","","","","","","",""]
t_stat_names = ["lineup","blocks","tovs","break","start_offensive_pt","offensive_ps","scored_us","scored_ps","score_us_ps","score_opp_ps","set_play_o","d_tactic","negative_hystory","positive_hystory","assistmen","scorer","tot_blocks","tot_tovs","ht_blocks","ht_tovs","tot_break_us","tot_break_opp","ht_break_us","ht_break_opp","final_score_us","final_score_opp","ht_score_us","ht_score_opp"]
t_stat_recurrency = ["pt","pt","pt","pt","pt","ps","pt","ps","ps","ps","pt","pt","pt","pt","ps","ps","gm","gm","gm","gm","gm","gm","gm","gm","gm","gm","gm","gm"] # lineup is per point, we still don't manage perfectly the injury subs
t_stat_dtype = ["[7]","int","int","bool","bool","bool","bool","bool","int","int","string","string","string","string","int","int","int","int","int","int","int","int","int","int","int","int","int","int"]
t_stat_owners = ["team"]*len(t_stat_names)
t_stat_important = ['!', '', '', '!', '!', '!', '!', '!', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] # [""]*len(t_stat_names)

# print(len(t_stat_short_names))
# print(len(t_stat_names))
# print(len(t_stat_recurrency))
# print(len(t_stat_dtype))
# print(len(t_stat_owners))

input_stats = {
    'short_names': stat_short_names+t_stat_short_names,
    'names': stat_names+t_stat_names,
    '!': stat_important+t_stat_important,
    'recurrency': stat_recurrency+t_stat_recurrency,
    'owners': stat_owners+t_stat_owners,
    'dtype': stat_dtype+t_stat_dtype
}
input_stats_df = pd.DataFrame(input_stats)
# print(input_stats_df.to_string())
# input_stats_df.to_csv(cleaned_data_dir / "def_stats.csv", index=False)




#For Advanced KPI (output_stats or something else) => 
# Ricerche = [,'Blocks linea D', 'Blocks/meteg D', 'Break seg D', 'Break/meteg D','TOV linea O', 'TOV/meteg O', 'Break sub O', 'Break/meteg O']
# def_kpi.csv
kpi_names = ["possessions_points","o_possessions_points","d_possessions_points","player_pm","scoring_impact","tov_recovery_impact","break_efficiency","impact_metric","possessions_points_X","o_possessions_points_X","d_possessions_points_X","team_pm_X","scoring_impact_X","tov_recovery_impact_X","break_efficiency_X","impact_metric_X"]
kpi_recurrency = ["","","","","","","","","","","","","","","",""] # Not on the single game but a weighted average for all of them
kpi_owners = ["player","player","player","player","player","player","player","player","team","team","team","team","team","team","team","team"] # ["player"]*len(kpi_names)
# kpi_recurrency = ["","","","","","","","","","","","","","","",""]
kpi_dtype = ["float"]*len(kpi_names) # ["float","float","float","float","float","","","","",""] # ["float"]*len(kpi_names) 
kpi_description = ["#pss_played_tot/#pts_played_tot","#pss_played_o/#pts_played_o","#pss_played_d/#pts_played_d","plus-minus=TODO: ...","pt where player in lineup #scored_us / #pss_played_o","pt where player in lineup, ps where tov_rec_ps=(start_offensive_pt && !offensive_ps): #(tov_rec_ps && !scored_ps) / #tov_rec_ps","pt where player in lineup, ps where break_chance_ps=(!start_offensive_ps && offensive_ps): #break / #break_chance_ps","scoring_impact+tov_recovery_impact+break_efficiency","","","","","","","",""]
kpi_stats_needed = [["pts_played_tot","pss_played_tot"],["pts_played_o","pss_played_o"],["pts_played_d","pss_played_d"],[],["lineup","scored_us","pss_played_o"],["lineup","start_offensive_pt","offensive_ps","scored_ps"],["lineup","start_offensive_ps","offensive_ps"],[],[],[],[],[],[],[],[],[]]

# print(len(kpi_names))
# print(len(kpi_owners))
# print(len(kpi_dtype))
# print(len(kpi_description))
# print(len(kpi_recurrency))
# print(len(kpi_stats_needed))

output_kpi = {
    'names': kpi_names,
    'owners': kpi_owners,
    'dtype': kpi_dtype,
    'description': kpi_description,
    'stats_needed': kpi_stats_needed
}
output_kpi_df = pd.DataFrame(output_kpi)
# print(output_kpi_df.to_string())
# output_kpi_df.to_csv(cleaned_data_dir / "def_kpi_df.csv", index=False)




# Games
#TODO: add columns: 
# - us_score
# - opp_score
# - us_ht_score
# - opp_ht_score 
dates = []
possible_stakes = ['quarti','semi','final', 'pool']
opponents = []
stakes = []
game_files = []
extra = 0
for (i, file) in enumerate(input_files):
    opponent = file.stem.split(" - ")[-1] #os.path.basename(file).split(" - ")[-1].split(".")[0]
    if opponent in ["Legenda Storia", "Tot"]:
        extra += 1
        continue
    else:
        game_files.append(file)
    for stake in possible_stakes:
        # NOTE: This works because stakes are all different names and different from the team names, otherwise manually
        if stake in opponent:
            opponents.append(opponent.split(stake)[0])
            stakes.append(stake)
            if stake in ['semi','final']:
                dates.append('29/09/24')
            elif stake == 'quarti':
                dates.append('28/09/24')
            break
    if opponent == 'BADSKID':
        opponents.append(opponent)
        dates.append('28/09/24')
        stakes.append('pool')
    elif len(stakes) == i-extra:
        opponents.append(opponent)
        dates.append('27/09/24')
        stakes.append('pool')


# Stats to add: ['pts_played_o', 'pts_played_d', 'pts_played_tot', 'tot_blocks', 'tot_tovs', 'ht_blocks', 'ht_tovs', 'tot_break_us', 'tot_break_opp', 'ht_break_us', 'ht_break_opp', 'final_score_us', 'final_score_opp', 'ht_score_us', 'ht_score_opp']
# DataFrame select column values
def_game_stats = input_stats_df[input_stats_df.recurrency == "gm"]
# print(def_game_stats)
# # DataFrame select column
# print(def_game_stats[["names","owners"]])
# # DataFrame column toList
# print(def_game_stats["names"].tolist())
# # DataFrame row toList
# print(def_game_stats.iloc[4,:].values.tolist())

def calc_gm_player():
    for (i, file) in enumerate(game_files):
        # Opponent:
        # opponents[i]
        pass


    pass

def calc_gm_team():
    #for BADSKID: from row that starts with "Tot", columns: 10,11
    #tot_blocks in 10,31 
    #tot_tovs in 11,31 
    #ht_blocks to_calc 
    #ht_tovs da to_calc

    pass


##BIG BIG NOTE: more important to start with the important stats, later we can recover the others.


gm_player = calc_gm_player()
gm_team = calc_gm_team()


# print(opponents)
# print(game_files)
games_df = pd.DataFrame({'opponent': opponents, 'stakes': stakes, 'date': dates})
# games = games[['opponent', 'stakes','date']] # 27,28,29/09/24
# print(opponents)
# print(stakes)
# print(dates)

# print(games_df.to_string())
# games_df.to_csv(cleaned_data_dir / "games.csv", index=False)






#points.csv
# Clutch Point Definition:
# 3pt left to the game point
# 2pt left to the ht_point
# AND
# 2pt or less difference (difficult to score more than 2 consecutive breaks)
def_pt_stats = input_stats_df[input_stats_df.recurrency == "pt"]
# print(def_pt_stats)

def calc_pt_player():
    pass

def calc_pt_team():
    pass


#possessions.csv
pt_player = calc_pt_player()
pt_team = calc_pt_team()












def_ps_stats = input_stats_df[input_stats_df.recurrency == "ps"]
# print(def_ps_stats)
def calc_ps_player():
    pass
def calc_ps_team():
    pass

ps = calc_ps_player()
ps = calc_ps_team()








##################
## Big Cleaning ##
##################
def extract_table3(df, tot_row):
    # Find the starting row of table 3 — first numeric value after 'Tot' row
    numeric_start = df.iloc[tot_row + 1:, 0].apply(lambda x: str(x).strip().isdigit())
    if not numeric_start.any():
        raise ValueError("Could not find the numeric start for table 3.")
    start_index = numeric_start[numeric_start].index[0]
    # Header row is the row right before numeric start
    header_row = df.loc[start_index - 1].dropna().tolist()
    # print(header_row)
    header_row = ["jersey_number","name"]+header_row
    # print(header_row)
    # Find the last column for table 3 — last occurrence of '(sec)' in the header row
    last_col_index = next((i for i, col in enumerate(header_row) if 'Blocks linea D' in str(col)), len(header_row) - 1) - 1
    # Find the last row — last numeric value in the first column
    numeric_values = df.loc[start_index:, 0].apply(lambda x: str(x).strip().isdigit())
    if not numeric_values.any():
        raise ValueError("Could not find the last numeric row for table 3.")
    end_index = numeric_values[numeric_values].index[-1]
    # Extract table 3
    table3 = df.loc[start_index:end_index, :last_col_index].reset_index(drop=True)
    # print(table3)
    # print(header_row[:last_col_index + 1])
    # print(table3.shape)
    # print(len(header_row[:last_col_index + 1]))
    table3.columns = header_row[:last_col_index + 1]
    table3 = table3[table3['name'].notna()]
    return table3


def extract_tables(filepath, last_point_id):
    df = pd.read_csv(filepath, header=None, skip_blank_lines=False)

    # Table 1: Starts from "In campo" row, ends when "In campo" column has empty cells
    in_campo_row = df[df.apply(lambda x: x.str.contains('In campo', na=False).any(), axis=1)].index[0]
    # in_campo_header = df.iloc[in_campo_row].dropna().tolist()
    in_campo_header = df.iloc[in_campo_row].tolist()
    # in_campo_header.remove("TODO")
    in_campo_header[0] = "special_line"

    table1_start = in_campo_row + 1
    table1_end = table1_start
    while table1_end < len(df) and pd.notna(df.iloc[table1_end, 3]):  # Column with "In campo" values
        table1_end += 1

    table1 = df.iloc[table1_start:table1_end].reset_index(drop=True)

    table1.columns = in_campo_header
    table1['In Campo'] = table1.iloc[:, 3:10].values.tolist()  # Group first 7 columns
    table1 = table1.drop(columns=table1.columns[2:10])  # Drop those 7 individual columns
    table1['point_id'] = range(last_point_id, last_point_id + len(table1))
    point_id = last_point_id + len(table1)
    print(table1)
    
    
    # print(len(in_campo_header))
    # # print(table1.num_columns)
    # print(len(in_campo_header))

    # print(table1)
    # print(in_campo_header)


    # Table 2: Row starting with "Tot"
    tot_row = df[df.iloc[:, 0].astype(str).str.contains('Tot', na=False)].index[0]
    table2 = pd.DataFrame([df.iloc[tot_row, 10:14].tolist()], columns=["tot_blocks", "tot_tovs", "tot_breaks_us", "tot_breaks_opp"])
    # print(table2)

    # # Table 3: Starts with first numeric value after Table 2
    # numeric_start = df[df.iloc[tot_row + 1:, 0].apply(lambda x: str(x).isdigit())].index[0]
    # header_row = numeric_start - 1
    # table3_end = numeric_start

    # while table3_end < len(df) and pd.to_numeric(df.iloc[table3_end, 0], errors='coerce').notna():
    #     table3_end += 1

    # table3 = df.iloc[numeric_start:table3_end].reset_index(drop=True)
    # print(table3)
    # table3.columns = df.iloc[header_row].dropna().tolist()



    # # Table 3: Starts from next numeric row, ends at last numeric row
    # numeric_start = df[df.iloc[tot_row + 1:, 0].apply(lambda x: str(x).strip().isdigit())].index[0]
    # header_row = numeric_start - 1
    # end_row_3 = df.loc[numeric_start:, 0].last_valid_index()

    # # End of table 3 based on last "(sec)" column
    # last_col_3 = df.loc[header_row].last_valid_index()
    # table_3 = df.loc[numeric_start:end_row_3, :last_col_3].reset_index(drop=True)
    # table_3.columns = df.loc[header_row, :last_col_3].tolist()
    table3 = extract_table3(df, tot_row)
    # print(table3)

    # Table 4: Last row after "TOT", sharing headers with Table 3
    tot_final_row = df[df.iloc[:, 1].astype(str).str.contains('TOT', na=False)].index[0]
    table4_columns = table3.columns[2:]
    print(len(df.iloc[tot_final_row, 2:].tolist()[:(len(table4_columns))]))
    print(df.iloc[tot_final_row, 2:].tolist()[:(len(table4_columns))])
    print(tot_final_row)
    print(len(table4_columns))
    # print(df.iloc[tot_final_row, 2:].tolist())
    table4 = pd.DataFrame([df.iloc[tot_final_row, 2:].tolist()[:(len(table4_columns))]], columns=table4_columns)
    print(table4)

    return table1, table2, table3, table4, point_id 



# Usage
tables = []
last_point_id = 0
for (i, file) in enumerate(game_files):
    table1, table2, table3, table4, last_point_id = extract_tables(file, last_point_id)
    table1['game_id'] = i
    table2['game_id'] = i
    table3['game_id'] = i
    table4['game_id'] = i
    tables.append(table1)
    tables.append(table2)
    tables.append(table3)
    tables.append(table4)

    # Display results
    print("Table 1:")
    print(table1.head())
    print("\nTable 2:")
    print(table2)
    print("\nTable 3:")
    print(table3.head())
    print("\nTable 4:")
    print(table4)
    # tables = extract_tables("Stats La FOTTA EUCF 2024 - Tot.csv")

points_team = pd.DataFrame(columns=tables[0].columns)
games_team = pd.DataFrame(columns=tables[1].columns+tables[1].columns)
points_player = pd.DataFrame(columns=tables[2].columns)
for i in range(0,4):
    points_team = pd.concat([points_team,tables[4*i]], ignore_index=True)

    # games_team = pd.concat([games_team,tables[4*i+1]], ignore_index=True)
    # games_team = pd.concat([games_team,tables[4*i+3]], ignore_index=True)

    # points_player = pd.concat([points_player,tables[4*i+2]], ignore_index=True)
    tables[4*i+1]
    tables[4*i+2]
    tables[4*i+3]

print(points_team)

# input_stats = {
#     'short_names': stat_short_names+t_stat_short_names,
#     'names': stat_names+t_stat_names,
#     '!': stat_important+t_stat_important,
#     'recurrency': stat_recurrency+t_stat_recurrency,
#     'owners': stat_owners+t_stat_owners,
#     'dtype': stat_dtype+t_stat_dtype
# }
# input_stats_df = pd.DataFrame(input_stats)







print("Data extraction complete. Structured files saved.")