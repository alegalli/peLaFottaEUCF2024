import os
import pandas as pd
import numpy as np
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
# player stats 24 elems
stat_short_names = ['S', 'D', 'HD', 'LT', 'T', 'P', 'Assist', 'Mete', 'Possesses Played Offence','Possesses Played Defence','Possesses Played Tot', 'Mete g att','Mete g dif','Mete g tot', 'Mga att (min)', '(sec)', 'Mga dif (min)', '(sec).1', 'Mga tot (min)', '(sec).2']
#not going to add info not strictly related to players AND points eg: clutch_point, special_line, break, scored_us, start_offensive_pt (shared in team_points, go find them there) # TODO: I think I can remove the time stats in player_points and put them in team_points, other than team_game
stat_names = ["super","block","help_defence","terrible_throw","throwaway","drop","assist","score","pss_played_o","pss_played_d","pss_played_tot","pts_played_o","pts_played_d","pts_played_tot","min_played_o","sec_played_o","min_played_d","sec_played_d","min_played_tot","sec_played_tot"] 
# stat_names_gm = ["super","block","help_defence","terrible_throw","throwaway","drop","assist","score","pss_played_o","pss_played_d","pss_played_tot","pts_played_o","pts_played_d","pts_played_tot","min_played_o","sec_played_o","min_played_d","sec_played_d","min_played_tot","sec_played_tot"]
stat_recurrency = ["pt","pt","pt","pt","pt","pt","ps","ps","pt","pt","pt","pt","pt","pt","pt","pt","pt","pt","pt","pt"] # ps=ps+pt+gm, pt=pt+gm, gm=gm
stat_dtype_pt = ["int","int","int","int","int","int","bool","bool","int","int","int","bool","bool","bool","int","int","int","int","int","int"] # 
stat_dtype_gm = ["int"]*len(stat_names) # 
stat_owners = ["player"]*len(stat_names)
stat_important = ['', '!', '!', '!', '!', '!', '', '', '', '', '!', '', '', '!', '', '', '', '', '', ''] # [""]*len(stat_names)
# print(len(kpi_names))
# print(len(stat_important))
# print(len(stat_important))
#team stats: 19 elems, Lineup.dtype = [], Break.dtype = bool
t_stat_short_names = ["","","","In Campo","Blocks","Tovs","Break","","A/D","","","","","","Schema A","Schema D","Storia neg","Storia pos","Assist","Meta","","","","","","","","","","","","","game_phase"]
t_stat_names = ["min_pt","sec_pt","special_line","lineup","blocks","tovs","break","clutch_point","start_offensive_pt","offensive_ps","scored_us","scored_ps","score_us_pt","score_opp_pt","set_play_o","d_tactic","negative_hystory","positive_hystory","assistmen","scorer","tot_blocks","tot_tovs","ht_blocks","ht_tovs","tot_break_us","tot_break_opp","ht_break_us","ht_break_opp","final_score_us","final_score_opp","ht_score_us","ht_score_opp","game_phase"]
# t_stat_names = ["min_pt","sec_pt","special_line","lineup","blocks","tovs","break","clutch_point","start_offensive_pt","offensive_ps","scored_us","scored_ps","score_us_pt","score_opp_pt","set_play_o","d_tactic","negative_hystory","positive_hystory","assistmen","scorer","tot_blocks","tot_tovs","ht_blocks","ht_tovs","tot_break_us","tot_break_opp","ht_break_us","ht_break_opp","final_score_us","final_score_opp","ht_score_us","ht_score_opp"]
t_stat_recurrency = ["pt","pt","pt","pt","pt","pt","pt","pt","pt","ps","pt","ps","pt","pt","pt","pt","pt","pt","ps","ps","gm","gm","gm","gm","gm","gm","gm","gm","gm","gm","gm","gm","gm"] # lineup is per point, we still don't manage perfectly the injury subs
t_stat_dtype = ["int","int","bool","[7]","int","int","bool","bool","bool","bool","bool","bool","int","int","string","string","string","string","int","int","int","int","int","int","int","int","int","int","int","int","int","int","string"]
t_stat_owners = ["team"]*len(t_stat_names)
t_stat_important = ['', '', '', '!', '', '', '!', '!', '!', '!', '!', '!', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '!'] # [""]*len(t_stat_names)

# print(len(stat_short_names))
# print(len(stat_names))
# print(len(stat_recurrency))
# print(len(stat_dtype_pt))
# print(len(stat_owners))
# print(len(stat_important))

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
    'dtype_ps': stat_dtype_pt+t_stat_dtype,
    'dtype_gm': stat_dtype_gm+t_stat_dtype
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








####################
## Big Collecting ##
####################
def extract_table3(df, tot_row):
    # Find the starting row of table 3 — first numeric value after 'Tot' row
    numeric_start = df.iloc[tot_row + 1:, 0].apply(lambda x: str(x).strip().isdigit())
    if not numeric_start.any():
        raise ValueError("Could not find the numeric start for table 3.")
    start_index = numeric_start[numeric_start].index[0]

    # Header row is the row right before numeric start
    header_row = df.loc[start_index - 1].dropna().tolist()
    # print(header_row)
    header_row = ["jersey_number","name"]+header_row[0:11]+["sec_played_o"]+[header_row[12]]+["sec_played_d"]+header_row[14:16]+["sec_played_tot"]
    # print(header_row) #14 16 19

    # last column
    last_col_index = next((i for i, col in enumerate(header_row) if 'Blocks linea D' in str(col)), len(header_row) - 1)
    # print(last_col_index)
    # last row
    numeric_values = df.loc[start_index:, 0].apply(lambda x: str(x).strip().isdigit())
    if not numeric_values.any():
        raise ValueError("Could not find the last numeric row for table 3.")
    end_index = numeric_values[numeric_values].index[-1]

    # Extract table 
    table3 = df.loc[start_index:end_index, :last_col_index].reset_index(drop=True)
    table3.columns = header_row[:last_col_index + 1]
    table3 = table3[table3['name'].notna()]


    # return data_table3, columns_table3
    return table3


def extract_tables(filepath, last_point_id):
    df = pd.read_csv(filepath, header=None, skip_blank_lines=False)

    # Table 1: Starts from "In campo" row, ends when "In campo" column has empty cells
    in_campo_row = df[df.apply(lambda x: x.str.contains('In campo', na=False).any(), axis=1)].index[0]
    # in_campo_header = df.iloc[in_campo_row].dropna().tolist()
    in_campo_header = df.iloc[in_campo_row].tolist()
    in_campo_header[0] = "special_line"

    table1_start = in_campo_row + 1
    table1_end = table1_start
    while table1_end < len(df) and pd.notna(df.iloc[table1_end, 3]):  # Column with "In campo" values
        table1_end += 1

    table1 = df.iloc[table1_start:table1_end].reset_index(drop=True)

    # TODO: in not collecting the min_pt && sec_pt for team_points. # Time not used for this project
    table1.columns = in_campo_header
    table1['In Campo'] = table1.iloc[:, 3:10].values.tolist()  # Group first 7 columns
    table1 = table1.drop(columns=table1.columns[2:10])  # Drop those 7 individual columns
    table1['point_id'] = range(last_point_id, last_point_id + len(table1))
    point_id = last_point_id + len(table1)
    # print(table1)

    # Table 2: Row starting with "Tot"
    tot_A_row = df[df.iloc[:, 0].astype(str).str.contains('Tot', na=False)].index[0]
    # TODO: trying to recover the data in a list instead of a DataFrame
    data_table2 = [df.iloc[tot_A_row, 10:14].tolist()]
    if (i==0):
        col_table2 = ["tot_blocks", "tot_tovs", "tot_breaks_us", "tot_breaks_opp"]
    table2 = pd.DataFrame([df.iloc[tot_A_row, 10:14].tolist()], columns=["tot_blocks", "tot_tovs", "tot_breaks_us", "tot_breaks_opp"])
    # print(table2)

    # # Table 3: Starts from next numeric row, ends at last numeric row
    table3 = extract_table3(df, tot_A_row)
    # print(table3)

    # Table 4: Last row after "TOT", sharing headers with Table 3
    tot_final_row = df[df.iloc[:, 1].astype(str).str.contains('TOT', na=False)].index[0]
    table4_columns = table3.columns[2:]
    # print(df.iloc[tot_final_row, 2:].tolist())
    table4 = pd.DataFrame([df.iloc[tot_final_row, 2:].tolist()[:(len(table4_columns))]], columns=table4_columns)
    # print(table4)

    # return data_table1, column_table1,data_table2, column_table2,data_table3, column_table3,data_table4, column_table4, point_id
    return table1, table2, table3, table4, point_id 



# Usage
tables1 = []
tables2 = []
tables3 = []
tables4 = []
last_point_id = 0
for (i, file) in enumerate(game_files):
    table1, table2, table3, table4, last_point_id = extract_tables(file, last_point_id)
    table1['game_id'] = i
    table3['game_id'] = i # concat tables2 & tables4
    table4['game_id'] = i
    tables1.append(table1)
    tables2.append(table2)
    tables3.append(table3)
    tables4.append(table4)

# # Display results
# print("Table 1:")
# # print(tables1)
# # print(tables1.head())
# print("\nTable 2:")
# print(tables2)
# print("\nTable 3:")
# print(tables3)
# # print(tables3.head())
# print("\nTable 4:")
# print(tables4)

points_team = pd.DataFrame(columns=tables1[0].columns)
# print(points_team)
print("games_team")
# TODO: I know that I should save the data on lists and then only later work with dfs, to redo
games_team = pd.concat([pd.concat(tables2, axis=0, ignore_index=True),pd.concat(tables4, axis=0, ignore_index=True)],axis=1)
points_team = pd.concat(tables1, axis=0, ignore_index=True)
print(points_team)

# points_player = pd.concat(tables3, axis=0, ignore_index=True)
games_player = pd.concat(tables3, axis=0, ignore_index=True)

# print(points_player)
# print(points_player.columns)
# games_df["game_id"] = range(1, len(games_df) + 1)
# points_team_df = tables1.merge(games_df[["game_id"]], on="game_id", how="left")
# points_player_df = tables3.merge(games_df[["game_id"]], on="game_id", how="left")
# print(points_team_df)
# print(points_player_df)
# print(games_team)
# print(points_team)
# print(points_player)
# print(games_team.columns[games_team.columns.duplicated()])

# print(def_game_stats)
# print(def_pt_stats)
# print()
pt_player_stats = input_stats_df[input_stats_df.recurrency == "pt"][input_stats_df.owners == "player"].names.reindex()
ps_player_stats = input_stats_df[input_stats_df.recurrency == "ps"][input_stats_df.owners == "player"].names.reindex()
gm_player_stats = input_stats_df[input_stats_df.recurrency == "gm"][input_stats_df.owners == "player"].names.reindex()
pt_team_stats = input_stats_df[input_stats_df.recurrency == "pt"][input_stats_df.owners == "team"].names.reindex()
ps_team_stats = input_stats_df[input_stats_df.recurrency == "ps"][input_stats_df.owners == "team"].names.reindex()
gm_team_stats = input_stats_df[input_stats_df.recurrency == "gm"][input_stats_df.owners == "team"].names.reindex()

# print(input_stats_df)


#################
# Points-Player #
#################
# Empty Points-Player columns TODO: recalc table from team_pt stats
# NOTE: do not initialize with NaN values, TODO: add possession_id
# points_player = points_player[list(points_player.columns[0:2])]
# print(pt_player_stats)
# points_player = pd.DataFrame({})
points_player_columns = ( #
    list(players_df.columns[0:2]) + # jersey_number, name
    list(input_stats_df[input_stats_df.owners == "player"].names.to_list()) + #all player stats
    ["point_id","game_id"]
)
# print(len(points_player_columns))
# print(points_player_columns)
# TODO: recalc from Points-Team


###############
# Points-Team #
###############
points_team.insert(0, 'min_pt', np.nan)
points_team.insert(1, 'sec_pt', np.nan)
in_campo = points_team.pop('In Campo')
points_team.insert(3, 'In Campo', in_campo)
points_team.insert(7, 'clutch_point', np.nan) #'') # TODO: calc clutch_point & scored_us
points_team.insert(9, 'scored_us', np.nan)
# print(points_team)
# print(points_team.columns)
# print(len(points_team.columns))
points_team.insert(10, 'score_us_pt', np.nan) #from ps_team_stats
points_team.insert(11, 'score_opp_pt', np.nan)
# print(points_team)
# print(points_team.columns)
# print(len(points_team.columns))
# # points_team.insert(18, 'assistmen', np.nan)
# points_team.insert(19, 'scorer', np.nan)
# print(points_team.columns)
# print(len(points_team.columns))
# print(pt_team_stats)
# print(len(pt_team_stats))
# print(ps_team_stats)
# print(len(ps_team_stats))
# print(points_team.shape)
print(pt_team_stats.to_list()+ps_team_stats.to_list()[2:]+points_team.columns.to_list()[18:])
print(len(pt_team_stats.to_list()+ps_team_stats.to_list()[2:]+points_team.columns.to_list()[18:]))
# team_ps
# "scored_us":bool,"scored_ps":bool,"score_us_ps":int,"score_opp_ps":int,

points_team.columns = pt_team_stats.to_list()+ps_team_stats.to_list()[2:]+points_team.columns.to_list()[18:]
print(points_team.columns)
print(points_team)

print(points_team[points_team['game_id'] == 4])
# # Calc scored_us
# scored_us = (not points_team['scored_us'].isna())
# points_team['scored_us'] = scored_us

# # Calc score_us_pt
# score_us_pt = (points_team['scored_us'])
# points_team['scored_us'] = scored_us

# # Calc score_opp_pt
# scored_us = (not points_team['scored_us'].isna())
# points_team['scored_us'] = scored_us
# Calc scored_us: True if 'scored_us' is not NaN
points_team['scored_us'] = points_team['scorer'].notna()

# Calc score_us_pt & score_opp_pt
points_team['score_us_pt'] = 0
points_team['score_opp_pt'] = 0

for i in range(len(points_team)):
    if i == 0 or points_team.loc[i, 'game_id'] != points_team.loc[i - 1, 'game_id']:
        # Start of a new game, reset scores
        prev_us_score = 0
        prev_opp_score = 0
    else:
        prev_us_score = points_team.loc[i - 1, 'score_us_pt']
        prev_opp_score = points_team.loc[i - 1, 'score_opp_pt']
    if points_team.loc[i, 'scored_us']:
        points_team.loc[i, 'score_us_pt'] = prev_us_score + 1
        points_team.loc[i, 'score_opp_pt'] = prev_opp_score
    else:
        points_team.loc[i, 'score_us_pt'] = prev_us_score
        points_team.loc[i, 'score_opp_pt'] = prev_opp_score + 1

print(points_team[points_team['game_id'] == 1][['game_id', 'scored_us', 'score_us_pt', 'score_opp_pt', 'scorer', 'game_phase']])


# print(points_team)







# Calculation Coposed Fields: clutch_point, scored_us
# Clutch Point Definition:
# 3pt left to the game point
# 2pt left to the ht_point
# AND
# 2pt or less difference (difficult to score more than 2 consecutive breaks)
# Index(['min_pt', 'sec_pt', 'special_line', 'lineup', 'blocks', 'tovs', 'break', 'clutch_point', 'start_offensive_pt', 'scored_us', 'set_play_o', 'd_tactic', 'negative_hystory', 'positive_hystory', 'assist', 'score', 'point_id', 'game_id'],
# NEED: games_team.final_score_us, games_team.final_score_opp, game_team.ht_score_us, game_team.ht_score_opp
# team_ps
# "scored_us":bool,"scored_ps":bool,"score_us_ps":int,"score_opp_ps":int,
# team_gm
# "tot_blocks","tot_tovs","ht_blocks","ht_tovs","tot_break_us","tot_break_opp","ht_break_us","ht_break_opp","final_score_us","final_score_opp","ht_score_us","ht_score_opp"

# Pseudo-code
# clutch_game_diff = 3  # 2 + 1
# clutch_ht_val = 3     # 2 + 1
# clutch_game_val = 4   # 3 + 1
# for gm in games_team:
#     max_ht_score = max(gm_ht_score_us, gm.ht_score_opp) 
#     max_final_score = max(gm.final_score_us, gm.final_score_opp)
#     for ps in points_team[points_team.game_id = gm].list():
#         if(abs(team_ps.score_us_ps - team_ps.score_opp_ps) < clutch_game_diff):
#             if((max_ht_score - team_ps.score_us_ps < clutch_ht_val) or (max_ht_score - team_ps.score_opp_ps < clutch_ht_val)):
#                 team_ps[gm][ps] = True
#                 pass
#             elif((max_final_score - team_ps.score_us_ps < clutch_game_val) or (max_final_score - team_ps.score_opp_ps < clutch_game_val)):
#                 team_ps[gm][ps] = True
#                 pass
#             else:
#                 team_ps[gm][ps] = False
# print(points_team)




#TODO: NEXT CLUTCH
# # # # # # # # clutch_game_diff = 3  # 2 + 1
# # # # # # # # clutch_ht_val = 3     # 2 + 1
# # # # # # # # clutch_game_val = 4   # 3 + 1
# # # # # # # # for game_id in games_team['game_id'].unique():
# # # # # # # #     game_points = points_team[points_team['game_id'] == game_id]
    
# # # # # # # #     # Get the highest scores in the game
# # # # # # # #     max_ht_score = max(game_points['score'].max(), 0)  # Max half-time score
# # # # # # # #     max_final_score = max(game_points['score'].max(), 0)  # Max final score

# # # # # # # #     for idx, point in game_points.iterrows():
# # # # # # # #         score_us = point['score']
# # # # # # # #         score_diff = abs(score_us - game_points['score'].shift().fillna(10))

# # # # # # # #         # Check if the point qualifies as "clutch"
# # # # # # # #         is_clutch = (
# # # # # # # #             (score_diff <= clutch_game_diff) and
# # # # # # # #             (
# # # # # # # #                 (max_ht_score - score_us < clutch_ht_val) or
# # # # # # # #                 (max_final_score - score_us < clutch_game_val)
# # # # # # # #             )
# # # # # # # #         )

# # # # # # # #         # Update the 'clutch_point' column
# # # # # # # #         points_team.at[idx, 'clutch_point'] = is_clutch








# print(games_team)
# print(games_team)




# print(points_team.columns)
# print("points_team")
# print(points_team.columns)
# print(games_player)
# print(games_team)



# # for (i, col) in enumerate(pt_player_stats):
#     points_player.insert(2+i, col, np.nan)

# print(points_player)# points_player

# print(points_player[list(points_player.columns[0:2]) + list(points_player.columns[10:])])
# # TODO: A different method to select 
# mete_g_att = points_player.pop('Mete g att')
# mete_g_dif = points_player.pop('Mete g dif')
# mete_g_tot = points_player.pop('Mete g tot')
# mga_att = points_player.pop('Mga att (min)')
# mga_dif = points_player.pop('Mga dif (min)')
# mga_tot = points_player.pop('Mga tot (min)')
# sga_att = points_player.pop('sec_played_o')
# sga_dif = points_player.pop('sec_played_d')
# sga_tot = points_player.pop('sec_played_tot')
# points_player.insert(10, 'pss_played_o', np.nan)
# points_player.insert(11, 'pss_played_d', np.nan)
# points_player.insert(12, 'pss_played_tot', np.nan)
# print(points_player.columns)
# print(games_player.columns)
# print(pt_player_stats)
# print(ps_player_stats)
# print(gm_player_stats)



# print(f"points_player.columns[0:2]: {points_player.columns[0:2]}")
# print(f"len(points_player.columns[0:2]): {len(points_player.columns[0:2])}")
# print(f"pt_player_stats.to_list()[0:6]: {pt_player_stats.to_list()[0:6]}")
# print(f"len(pt_player_stats.to_list()[0:6]): {len(pt_player_stats.to_list()[0:6])}")
# print(f"ps_player_stats.to_list(): {ps_player_stats.to_list()}")
# print(f"len(ps_player_stats.to_list()): {len(ps_player_stats.to_list())}")
# print(f"points_player.columns[10:]: {points_player.columns[10:]}")
# print(f"len(points_player.columns[10:]): {len(points_player.columns[10:])}")
# points_player.columns = ( #TODO: to edit
#     list(points_player.columns[0:2]) + # jersey_number, name
#     list(pt_player_stats.to_list()[0:6]) + #TODO: change base stats to game
#     list(ps_player_stats.to_list()) + # assist, score
#     list(points_player.columns[10:]) # pt_o,d,tot:bool, mga_o,d,tot, sga_o,d,tot ps_o,d,tot, game_id
# )

# TODO: easy compile games_player
games_player.columns = (
    list(games_player.columns[0:2]) + # jersey_number, name
    list(pt_player_stats.to_list()[0:6]) + #TODO: change base stats to game
    list(ps_player_stats.to_list()) + # assist, score
    list(games_player.columns[10:]) # pss_o,d,tot, game_id
)
# points_player.columns = points_player.columns[0:2]+pt_player_stats.to_list()[0:6]+ps_player_stats.to_list()+points_player.columns[10:] # TODO: to calc possessions_played and move points_played into game stats
# print(points_player)



# print(points_team)# input_stats = {
#     'short_names': stat_short_names+t_stat_short_names,
#     'names': stat_names+t_stat_names,
#     '!': stat_important+t_stat_important,
#     'recurrency': stat_recurrency+t_stat_recurrency,
#     'owners': stat_owners+t_stat_owners,
#     'dtype': stat_dtype+t_stat_dtype
# }
# input_stats_df = pd.DataFrame(input_stats)







print("Data extraction complete. Structured files saved.")