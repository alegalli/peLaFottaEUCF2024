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


# Games
dates = []
possible_stakes = ['quarti','semi','final', 'pool']
opponents = []
stakes = []
extra = 0
for (i, file) in enumerate(input_files):
    opponent = file.stem.split(" - ")[-1] #os.path.basename(file).split(" - ")[-1].split(".")[0]
    if opponent in ["Legenda Storia", "Tot"]:
        extra += 1
        continue
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
    if len(stakes) == i-extra:
        opponents.append(opponent)
        dates.append('27/09/24')
        stakes.append('pool')
    if opponent == 'BADSKID':
        dates[-1] = '28/09/24' #dates.append('28/09/24')


games_df = pd.DataFrame({'opponent': opponents, 'stakes':stakes, 'date': dates})
# games = games[['opponent', 'stakes','date']] # 27,28,29/09/24
print(opponents)
print(stakes)
print(dates)

print(games_df.to_string())
games_df.to_csv(cleaned_data_dir / "games.csv", index=False)

# print(os.path.basename(file).split(" - ")[-1].split(".")[0])




    # df = pd.read_csv(file, header=None)
    # print(df.to_string())

# games = games[['opponent', 'stakes','date']] # 27,28,29/09/24
# print(games.to_string())





# df = pd.read_csv(legend, header=None)

# data = df[df.iloc[:, 0] == "Laffi"].index[0] + 1

# for file in input_files:
#     df = pd.read_csv(file, header=None)

#     print(df.to_string())

    # # Identify sections (tables within the CSV)
    # game_info_start = df[df.iloc[:, 0] == "Laffi"].index[0] + 1
    # # player_stats_start = df[df.iloc[:, 0] == "Player"].index[0] + 1

    # print(type(game_info_start))




    # # Extract game-level stats
    # game_stats = df.iloc[game_info_start:game_info_start + 1].values[0]
    # games.append({
    #     "game_id": os.path.basename(file),
    #     "home_score": game_stats[1],
    #     "opp_score": game_stats[2],
    #     "tot_tovs": game_stats[3],
    #     "tot_blocks": game_stats[4],
    #     "tot_breaks_scored": game_stats[5],
    #     "tot_break_opp": game_stats[6]
    # })

#     # Extract player stats
#     player_data = df.iloc[player_stats_start:].dropna(how='all')
#     for _, row in player_data.iterrows():
#         players.append({
#             "game_id": os.path.basename(file),
#             "player": row[0],
#             "S": row[1], "D": row[2], "HD": row[3], "LT": row[4], "T": row[5],
#             "P": row[6], "Assist": row[7], "Mete": row[8], "Mete_g_att": row[9],
#             "Mete_g_dif": row[10], "Mga_att": row[11], "Mga_dif": row[12],
#             "Mga_tot": row[13]
#         })

# # Convert to DataFrame
# games_df = pd.DataFrame(games)
# players_df = pd.DataFrame(players)
# # points_df = pd.DataFrame(points)
# # events_df = pd.DataFrame(events)

# # Save structured tables
# games_df.to_csv("C:\\Users\\Alessandro\\Documents\\ale\\valuationLaFotta2024\\cleaned_data\\games.csv", index=False)
# players_df.to_csv("C:\\Users\\Alessandro\\Documents\\ale\\valuationLaFotta2024\\cleaned_data\\players.csv", index=False)
# # points_df.to_csv("C:\\Users\\Alessandro\\Documents\\ale\\valuationLaFotta2024\\cleaned_data\\points.csv", index=False)
# # events_df.to_csv("C:\\Users\\Alessandro\\Documents\\ale\\valuationLaFotta2024\\cleaned_data\\events.csv", index=False)

print("Data extraction complete. Structured files saved.")