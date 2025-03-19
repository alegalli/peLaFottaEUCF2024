import os
import pandas as pd
import numpy as np
from pathlib import Path

# Define input file pattern (adjust path if needed)
data_dir = Path('data')
cleaned_data_dir = Path('cleaned_data')
input_files = list(data_dir.glob("Stats La FOTTA EUCF 2024 - *.csv"))
tot = "data/Stats La FOTTA EUCF 2024 - Tot.csv"
# tot = data_dir.glob("Stats La FOTTA EUCF 2024 - Tot.csv")

# Players
players = pd.read_csv(tot, usecols=[0,1])
players.rename(columns={'Unnamed: 0': 'jersey_number', 'Unnamed: 1': 'name'}, inplace=True)
players['team'] = 'BFD La Fotta'
players.dropna(inplace=True)
players['jersey_number'] = players['jersey_number'].astype('int')

# for player_idx, player in players.iterrows():
players['player_id'] = list(range(0, len(players['name']))) #[player_idx] = player_idx
players['role'] = ['H']*4 + ['Y']*4 + ['C']*3 + ['H']*5 + ['Y']*6 + ['C']*3
players['line'] = ['O']*11 + ['D']*14
# print(players)
players = players[['player_id','name', 'jersey_number','role','line','team']]
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

#team stats: 19 elems, Lineup.dtype = [], Break.dtype = bool
t_stat_short_names = ["","","","In Campo","Blocks","Tovs","Break","","A/D","","","","","","Schema A","Schema D","Storia neg","Storia pos","Assist","Meta","","","","","","","","","","","","","game_phase"]
t_stat_names = ["min_pt","sec_pt","special_line","lineup","blocks","tovs","break","clutch_point","start_offensive_pt","offensive_ps","scored_us","scored_ps","score_us_pt","score_opp_pt","set_play_o","d_tactic","negative_hystory","positive_hystory","assistmen","scorer","tot_blocks","tot_tovs","ht_blocks","ht_tovs","tot_break_us","tot_break_opp","ht_break_us","ht_break_opp","final_score_us","final_score_opp","ht_score_us","ht_score_opp","game_phase"]
t_stat_recurrency = ["pt","pt","pt","pt","pt","pt","pt","pt","pt","ps","pt","ps","pt","pt","pt","pt","pt","pt","ps","ps","gm","gm","gm","gm","gm","gm","gm","gm","gm","gm","gm","gm","gm"] # lineup is per point, we still don't manage perfectly the injury subs
t_stat_dtype = ["int","int","bool","[7]","int","int","bool","bool","bool","bool","bool","bool","int","int","string","string","string","string","int","int","int","int","int","int","int","int","int","int","int","int","int","int","string"]
t_stat_owners = ["team"]*len(t_stat_names)
t_stat_important = ['', '', '', '!', '', '', '!', '!', '!', '!', '!', '!', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '!'] # [""]*len(t_stat_names)

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
input_stats_df.to_csv(cleaned_data_dir / "def_stats.csv", index=False)




#For Advanced KPI (output_stats or something else) =>
# Ricerche = [,'Blocks linea D', 'Blocks/meteg D', 'Break seg D', 'Break/meteg D','TOV linea O', 'TOV/meteg O', 'Break sub O', 'Break/meteg O']
# def_kpi.csv
kpi_names = ["possessions_points","o_possessions_points","d_possessions_points","player_pm","scoring_impact","tov_recovery_impact","break_efficiency","impact_metric","possessions_points_X","o_possessions_points_X","d_possessions_points_X","team_pm_X","scoring_impact_X","tov_recovery_impact_X","break_efficiency_X","impact_metric_X"]
kpi_recurrency = ["","","","","","","","","","","","","","","",""] # Not on the single game but a weighted average for all of them
kpi_owners = ["player","player","player","player","player","player","player","player","team","team","team","team","team","team","team","team"] # ["player"]*len(kpi_names)
# kpi_recurrency = ["","","","","","","","","","","","","","","",""]
kpi_dtype = ["float"]*len(kpi_names) # ["float","float","float","float","float","","","","",""] # ["float"]*len(kpi_names)
kpi_description = ["#pss_played_tot/#pts_played_tot","#pss_played_o/#pts_played_o","#pss_played_d/#pts_played_d","plus-minus=TODO: ...","pt where player in lineup #scored_us / #pss_played_o","pt where player in lineup, ps where tov_rec_ps=(start_offensive_pt && !offensive_ps): #(tov_rec_ps && !scored_ps) / #tov_rec_ps","pt where player in lineup, ps where break_chance_ps=(!start_offensive_ps && offensive_ps): #break / #break_chance_ps","scoring_impact+tov_recovery_impact+break_efficiency, to weight considering pts_played_o, pts_played_d","","","","","","","",""]
kpi_stats_needed = [["pts_played_tot","pss_played_tot"],["pts_played_o","pss_played_o"],["pts_played_d","pss_played_d"],[],["lineup","scored_us","pss_played_o"],["lineup","start_offensive_pt","offensive_ps","scored_ps"],["lineup","start_offensive_ps","offensive_ps","break"],[],[],[],[],[],[],[],[],[]]

output_kpi = {
    'names': kpi_names,
    'owners': kpi_owners,
    'dtype': kpi_dtype,
    'description': kpi_description,
    'stats_needed': kpi_stats_needed
}
output_kpi_df = pd.DataFrame(output_kpi)
# print(output_kpi_df.to_string())
output_kpi_df.to_csv(cleaned_data_dir / "def_kpi.csv", index=False)


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


games_df = pd.DataFrame({'opponent': opponents, 'stakes': stakes, 'date': dates})
games_df.to_csv(cleaned_data_dir / "games.csv", index=False)


#points.csv
# Clutch Point Definition:
# 3pt left to the game point
# 2pt left to the ht_point
# AND
# 2pt or less difference (difficult to score more than 2 consecutive breaks)
def_pt_stats = input_stats_df[input_stats_df.recurrency == "pt"]
# print(def_pt_stats)
def_ps_stats = input_stats_df[input_stats_df.recurrency == "ps"]
# print(def_ps_stats)


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
    # # TODO: trying to recover the data in a list instead of a DataFrame
    # data_table2 = [df.iloc[tot_A_row, 10:14].tolist()]
    # if (i==0):
    #     col_table2 = ["tot_blocks", "tot_tovs", "tot_breaks_us", "tot_breaks_opp"]
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

points_team = pd.DataFrame(columns=tables1[0].columns)
print("games_team")
# TODO: I know that I should save the data on lists and then only later work with dfs, to redo
games_team = pd.concat([pd.concat(tables2, axis=0, ignore_index=True),pd.concat(tables4, axis=0, ignore_index=True)],axis=1)
points_team = pd.concat(tables1, axis=0, ignore_index=True)
print(points_team)

games_player = pd.concat(tables3, axis=0, ignore_index=True)

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
points_player_columns = ( #
    list(players_df.columns[0:3]) + # jersey_number, name, role
    list(input_stats_df[input_stats_df.owners == "player"].names.to_list()) + #all player stats
    ["point_id","game_id"]
)
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
points_team.insert(10, 'score_us_pt', np.nan) #from ps_team_stats
points_team.insert(11, 'score_opp_pt', np.nan)

points_team.columns = pt_team_stats.to_list()+ps_team_stats.to_list()[2:]+points_team.columns.to_list()[18:]
# print(points_team.columns)
# print(points_team)

# print(points_team[points_team['game_id'] == 4])
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
    # Calc ht_score_us
    # Calc ht_score_opp
    # Calc final_score_us
    # Calc final_score_opp
    if points_team.loc[i, 'game_phase'] == 'ht': #no need for int() find another way to cast to int
        games_team.loc[points_team.loc[i, 'game_id'], 'ht_score_us'] = int(points_team.loc[i, 'score_us_pt'])
        games_team.loc[points_team.loc[i, 'game_id'], 'ht_score_opp'] = int(points_team.loc[i, 'score_opp_pt'])
    elif points_team.loc[i, 'game_phase'] == 'end':
        games_team.loc[points_team.loc[i, 'game_id'], 'final_score_us'] = int(points_team.loc[i, 'score_us_pt'])
        games_team.loc[points_team.loc[i, 'game_id'], 'final_score_opp'] = int(points_team.loc[i, 'score_opp_pt'])
    # Calc BOOLS
    # Calc start_offensive_pt
    if points_team.loc[i, 'start_offensive_pt'] == 'A' or points_team.loc[i, 'start_offensive_pt'] == 'a':
        points_team.loc[i, 'start_offensive_pt'] = True
    elif points_team.loc[i, 'start_offensive_pt'] == 'D' or points_team.loc[i, 'start_offensive_pt'] == 'd':
        points_team.loc[i, 'start_offensive_pt'] = False
    else:
        points_team.loc[i, 'start_offensive_pt'] = np.NaN
    # Calc break
    if points_team.loc[i, 'break'] == 'BREAK':
        points_team.loc[i, 'break'] = True
    else:
        points_team.loc[i, 'break'] = False
# print(points_team)


##########
# Points-Player
##########
# Note: to save db space we save only possession where a specific player has played. So if for a certain possession_id and player_id there is no record, it means that that player didn't played that possession => pt_played_tot
points_player = pd.DataFrame(columns=['point_id', 'player_id', 'name', 'role', 'line', 'pss_played_o', 'pss_played_d', 'pss_played_tot', 'pt_played_o', 'pt_played_d', 'pt_played_tot'])

# Loop through points_team
for point_idx, point in points_team.iterrows():
    lineup = point['lineup']  # This is already a list of integers
    if pd.isna(point['blocks']):
        point['blocks'] = 0
    if pd.isna(point['tovs']):
        point['tovs'] = 0
    # if point_idx == 10:
        # print(points_player)
    # Loop through players_df
    for player_idx, player in players_df.iterrows():
        jersey_number = player['jersey_number']  # This is also an integer

        # Check if the player is in the lineup
        if str(jersey_number) in lineup:
            # Calculate offensive and defensive stats
            pss_player_o = int(point['blocks']) + int(1 if point['start_offensive_pt'] else 0)
            pss_player_d = int(point['tovs']) + int((0 if point['start_offensive_pt'] else 1))
            pss_player_tot = int(point['blocks']) + int(point['tovs']) + 1

            # Determine if player played on offense or defense
            pt_played_o = point['start_offensive_pt']
            pt_played_d = not point['start_offensive_pt']
            pt_played_tot = True

            # Append the row to points_player
            points_player = pd.concat([points_player, pd.DataFrame([{
                'point_id': point['point_id'],
                'player_id': player['player_id'],  # <- using actual player_id now
                'name': player['name'],
                'role': player['role'],
                'line': player['line'],
                'pss_played_o': pss_player_o,
                'pss_played_d': pss_player_d,
                'pss_played_tot': pss_player_tot,
                'pt_played_o': pt_played_o,
                'pt_played_d': pt_played_d,
                'pt_played_tot': pt_played_tot
            }])], ignore_index=True)

print("points_player")

points_player.to_csv(cleaned_data_dir / "points_player.csv", index=False)
# print(input_stats_df)


# Calculation Composed Fields: clutch_point, scored_us
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
#TODO: NEXT CLUTCH
clutch_game_diff = 3  # 2 + 1
clutch_ht_val = 3     # 2 + 1
clutch_game_val = 4   # 3 + 1
for game_id in games_team['game_id'].unique():
    # gt = games_team[games_team['game_id'] == game_id]
    game_points = points_team[points_team['game_id'] == game_id]

    # Get the highest scores in the game
    max_ht_score = max(games_team.loc[game_id,'ht_score_us'], games_team.loc[game_id,'ht_score_opp'])  # Max half-time score
    min_ht_score = min(games_team.loc[game_id,'ht_score_us'], games_team.loc[game_id,'ht_score_opp'])  # Max half-time score
    max_final_score = max(games_team.loc[game_id,'final_score_us'], games_team.loc[game_id,'final_score_opp'])  # Max final score

    for idx, point in game_points.iterrows():
        max_score = max(point['score_us_pt'], point['score_opp_pt'])
        min_score = min(point['score_us_pt'], point['score_opp_pt'])
        score_diff = abs(point['score_us_pt'] - point['score_opp_pt'])

        # Check if the point qualifies as "clutch"
        is_clutch = (
            (score_diff <= clutch_game_diff) and
            (
                ((max_ht_score - max_score < clutch_ht_val) and (max_score <= max_ht_score) and (min_score <= min_ht_score)) or
                (max_final_score - max_score < clutch_game_val)
            )
        )
        points_team.loc[idx, 'clutch_point'] = is_clutch

# print(points_team)
# print(points_team[points_team['point_id'] > 120])
points_team.to_csv(cleaned_data_dir / "points_team.csv", index=False)




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
games_player = players_df[["name","role","line"]].merge(games_player.merge(players_df[["name","player_id"]], how="outer", on=["name"]), how="outer", on=["name"])
print(games_player)
# games_player.columns = (
#     list(games_player.columns[0:2]) + # jersey_number, name, role
#     [] +
#     list(pt_player_stats.to_list()[0:6]) + #TODO: change base stats to game
#     list(ps_player_stats.to_list()) + # assist, score
#     list(games_player.columns[10:]) # pss_o,d,tot, game_id
# )
games_player.to_csv(cleaned_data_dir / "games_player.csv", index=False)
print(games_player)

print('points_player_clutch')
points_player_clutch = points_player.merge(points_team)
points_player_clutch = points_player_clutch[points_player_clutch['clutch_point'] == True][points_player.columns].reset_index(drop=True)
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











###################
# KPI Computation #
###################

# print(output_kpi_df)
# possessions_points =
# o_possessions_points =
# d_possessions_points =

# 1) The first input to loc command is the filter for the index and then the second is the column.
# df.loc[df['Year'] == '2019', 'Arrived'].sum()
# 2) Another approach here, in case you wanted to have the sum for every year, would be to use the groupby operation:
# per_year = df.groupby('Year')['Arrived'].sum()
# This would give you a series, and you could then see the value for 2019 specifically with:
# per_year['2019']

# Safe division function
def safe_divide(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0

def calc_possessions(grouped, pt_pl):
    # Create the new DataFrame with calculated possession points per player
    # Usare le lambda commentate solo per selezionare i pss giocanti in quel punto o/d
    possession_points_per_player = grouped.agg({
        'pss_played_o': 'sum',#lambda x: x[pt_pl.loc[x.index, 'pt_played_o'] == True].sum(),
        'pt_played_o': 'sum',
        'pss_played_d': 'sum',#lambda x: x[pt_pl.loc[x.index, 'pt_played_d'] == True].sum(),
        'pt_played_d': 'sum',
        'pss_played_tot': lambda x: x[pt_pl.loc[x.index, 'pt_played_tot'] == True].sum(),
        'pt_played_tot': 'sum'
    }).reset_index()

    # TOCOMPLETE 17/03/2025
    #     possession_points_per_player = grouped.agg({
    #     'pss_played_o_o': lambda x: x[x['start_offensive_pt'] == True].sum(),
    #     'pss_played_o_d': lambda x: x[x['start_offensive_pt'] == False].sum(),
    #     # 'pss_played_o_d': 'sum',#lambda x: x[pt_pl.loc[x.index, 'pt_played_o'] == True].sum(),
    #     'pt_played_o': 'sum',
    #     'pss_played_d_o': lambda x: x[x['start_offensive_pt'] == True].sum(),
    #     'pss_played_d_d': lambda x: x[x['start_offensive_pt'] == False].sum(),
    #     # 'pss_played_d_o': 'sum',#lambda x: x[pt_pl.loc[x.index, 'pt_played_o'] == True].sum(),
    #     'pt_played_d': 'sum',
    #     'pss_played_tot': lambda x: x[pt_pl.loc[x.index, 'pt_played_tot'] == True].sum(),
    #     'pt_played_tot': 'sum'
    # }).reset_index()




    print("possession_points_per_player")
    print(possession_points_per_player)
    # Calculate possession points
    # tov_recovery_impact = grouped.apply(
    #     lambda g: (
    #         (g.loc[g['start_offensive_pt'] == True, 'pss_played_d'].sum() -
    #         g.loc[g['start_offensive_pt'] == True, 'break'].sum()) /
    #         g.loc[g['start_offensive_pt'] == True, 'pss_played_d'].sum()
    #     ) if g.loc[g['start_offensive_pt'] == True, 'pss_played_d'].sum() != 0 else np.nan
    # )
    possession_points_per_player['o_possessions_points'] = possession_points_per_player.apply(
        lambda row: safe_divide(row['pss_played_o'], row['pt_played_o']), axis=1
    )
    possession_points_per_player['o_possessions'] = possession_points_per_player.apply(
        lambda row: row['pss_played_o'], axis=1
    )
    possession_points_per_player['d_possessions_points'] = possession_points_per_player.apply(
        lambda row: safe_divide(row['pss_played_d'], row['pt_played_d']), axis=1
    )
    possession_points_per_player['d_possessions'] = possession_points_per_player.apply(
        lambda row: row['pss_played_d'], axis=1
    )
    possession_points_per_player['tot_possessions_points'] = possession_points_per_player.apply(
        lambda row: safe_divide(row['pss_played_tot'], row['pt_played_tot']), axis=1
    )
    possession_points_per_player['tot_possessions'] = possession_points_per_player.apply(
        lambda row: row['pss_played_tot'], axis=1
    )
    # Drop the intermediate columns if you don’t need them
    return possession_points_per_player[
        ['player_id', 'name', 'pt_played_o', 'o_possessions_points', 'o_possessions', 'pt_played_d', 'd_possessions_points', 'd_possessions', 'pt_played_tot', 'tot_possessions_points', 'tot_possessions',]
    ]

# Grouping the data without summing yet
grouped = points_player.groupby(['player_id', 'name'])
possession_points_per_player = calc_possessions(grouped, points_player)
# print(possession_points_per_player)

kpi_df = possession_points_per_player.merge(players_df)
# # # # kpi_df.to_csv(cleaned_data_dir / "kpi.csv", index=False)


# Grouping the data without summing yet
clutch_grouped = points_player_clutch.groupby(['player_id', 'name'])
clutch_possession_points_per_player = calc_possessions(clutch_grouped, points_player_clutch)
print(clutch_possession_points_per_player)

clutch_kpi_df = clutch_possession_points_per_player.merge(players_df)
# # # # clutch_kpi_df.to_csv(cleaned_data_dir / "clutch_kpi.csv", index=False)
print('clutch poss')
# print(clutch_kpi_df)
# print(kpi_df)

# print(points_player)
# print(points_player_clutch)


# Other KPIs:
# scoring_impact =lineup, scored_us, pss_played_o # no need to check pss_played_o>0 because I'm not counting the points played
# when in lineup:
#   scored_us.sum() / pss_played_o.sum()
# tov_recovery_impact =lineup, start_offensive_pt, tovs, blocks
# break_efficiency = lineup, start_offensive_pt, tovs, blocks, break
# impact_metric = #TOBEDEFINED
def calc_kpi(grouped):
    # Calculate scoring_impact: total points scored / total offensive possessions played
    scoring_impact = grouped.apply(
        lambda g: g['scored_us'].sum() / g['pss_played_o'].sum()
        if g['pss_played_o'].sum() != 0 else np.nan
    )

    # Calculate block_creation: blocks / pss_played_d
    block_creation = grouped.apply(
        lambda g: g['blocks'].sum() / g['pss_played_d'].sum() #toadd: blocks
        if g['pss_played_d'].sum() != 0 else np.nan
    )

    # Calculate tov_recovery_impact: (tov_rec_ps - taken_breaks) / tov_rec_ps
    tov_recovery_impact = grouped.apply(
        lambda g: (
            (g.loc[g['start_offensive_pt'] == True, 'pss_played_d'].sum() -
            g.loc[g['start_offensive_pt'] == True, 'break'].sum()) /
            g.loc[g['start_offensive_pt'] == True, 'pss_played_d'].sum()
        ) if g.loc[g['start_offensive_pt'] == True, 'pss_played_d'].sum() != 0 else np.nan
    )

    # Calculate break_efficiency: scored_breaks / break_chances
    break_efficiency = grouped.apply(
        lambda g: (
            g.loc[g['start_offensive_pt'] == False, 'break'].sum() /
            g.loc[g['start_offensive_pt'] == False, 'pss_played_o'].sum()
        ) if g.loc[g['start_offensive_pt'] == False, 'pss_played_o'].sum() != 0 else np.nan
    )
    return pd.DataFrame({
        'player_id': scoring_impact.index,
        'scoring_impact': scoring_impact.values,
        'block_creation': block_creation.values,
        'tov_recovery_impact': tov_recovery_impact.values,
        'break_efficiency': break_efficiency.values
    }).reset_index(drop=True)



# print(points_player.merge(players, how="inner").merge(points_team, how="outer", on="point_id")[['point_id', 'player_id', 'name', 'jersey_number', 'lineup', 'scored_us', 'pss_played_o', 'pss_played_d', 'start_offensive_pt', 'break']])
pt_pl_tm_kpi = points_player.merge(points_team, how="outer", on="point_id")[['point_id', 'player_id', 'name', 'role', 'line', 'lineup', 'scored_us', 'pss_played_o', 'pss_played_d', 'start_offensive_pt', 'break', 'blocks', 'tovs']]
pt_pl_tm_kpi['blocks'] = pt_pl_tm_kpi['blocks'].fillna(0).astype('uint8')
pt_pl_tm_kpi['tovs'] = pt_pl_tm_kpi['tovs'].fillna(0).astype('uint8')
# print(pt_pl_tm_kpi.dtypes)
grouped = pt_pl_tm_kpi.groupby('player_id', group_keys=False)
kpi_df2 = calc_kpi(grouped)

kpi_df = players.merge(kpi_df).merge(kpi_df2)
# print(kpi_df)

kpi_df.to_csv(cleaned_data_dir / "kpi.csv", index=False)
# print('kpi')


clutch_pt_pl_tm_kpi = points_player_clutch.merge(points_team, how="outer", on="point_id")[['point_id', 'player_id', 'name', 'role', 'line', 'lineup', 'scored_us', 'pss_played_o', 'pss_played_d', 'start_offensive_pt', 'break', 'blocks', 'tovs']].dropna(subset=['player_id','name'])
clutch_pt_pl_tm_kpi['blocks'] = clutch_pt_pl_tm_kpi['blocks'].fillna(0).astype('uint8')
clutch_pt_pl_tm_kpi['tovs'] = clutch_pt_pl_tm_kpi['tovs'].fillna(0).astype('uint8')
# print(clutch_pt_pl_tm_kpi.dtypes)
clutch_grouped = clutch_pt_pl_tm_kpi.groupby('player_id', group_keys=False)
clutch_kpi_df2 = calc_kpi(clutch_grouped)

# print('clutch kpi')
clutch_kpi_df = players.merge(clutch_kpi_df).merge(clutch_kpi_df2)
print(clutch_kpi_df)
# print(kpi_df)
clutch_kpi_df.to_csv(cleaned_data_dir / "clutch_kpi.csv", index=False)

print(points_player[points_player['name'] == "Martin"])

print("Data extraction complete. Structured files saved.")