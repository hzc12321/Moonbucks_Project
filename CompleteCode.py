import pandas as pd
import requests
import numpy
import numpy as np
import csv
import random
from os.path import exists
from sklearn import preprocessing
from words import goodwords, badwords, stopwords
import plotly.express as px
import matplotlib.pyplot as plt


# Problem 1
def filter_stopwords(text=[]):
    return list(filter(lambda t: t not in stopwords, text))


def word_freq(text=[]):
    return sorted(list(set(zip(text, list(map(lambda t: text.count(t), text))))), key=lambda x: x[1], reverse=True)


def sort_words(text=[]):
    good = []
    bad = []
    neutral = []

    for t in text:
        if t in goodwords:
            good.append(t)
        elif t in badwords:
            bad.append(t)
        else:
            neutral.append(t)

    return good, bad, neutral


def count_words(text=[]):
    return sum(list(map(lambda t: t[1], text)))


# Plot Graph
def plotChart(country):
    df_for_plotting = pd.read_csv(f'csv/{country}.csv')

    plotGoodWords(country)
    plotBadWords(country)
    plotWordCountPerCountryArticle(df_for_plotting.head(5))


def plotBadWords(state):
    filename = 'csv/bad-' + state + '.csv'
    df = pd.read_csv(filename)

    # each state
    countries = {
        "AE": "United Arab Emirates",
        "MY": "Malaysia",
        "SG": "Singapore",
        "US": "United States",
        "UK": "United Kingdom"
    }
    title = "Frequency of Bad Words for " + countries[state]

    fig = px.bar(df, y='Frequency', x='Word', text_auto='.2d', title=title)
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.show()


def plotGoodWords(state):
    filename = 'csv/good-' + state + '.csv'
    df = pd.read_csv(filename)

    # each state
    countries = {
        "AE": "United Arab Emirates",
        "MY": "Malaysia",
        "SG": "Singapore",
        "US": "United States",
        "UK": "United Kingdom"
    }
    title = "Frequency of Good Words for " + countries[state]

    fig = px.bar(df, y='Frequency', x='Word', text_auto='.2d', title=title)
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.show()


def plotWordCountPerCountryArticle(df):
    x = np.arange(5)
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, df['positive_word'].tolist(), width, label='Positive Words')
    rects2 = ax.bar(x + width / 2, df['negative_word'].tolist(), width, label='Negative Words')

    ax.set_ylabel('Words Count')
    ax.set_title('Number of Word Count for 5 articles per country')
    ax.set_xticks(x, df['Article'].tolist())
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()


def plotOverview():
    Articles = ["AE", "MY", "SG", "UK", "US"]
    positive = []
    negative = []
    for i in Articles:
        df = pd.read_csv(f'csv/{i}.csv')
        positive.append(df['positive_word'].iloc[5])
        negative.append(df['negative_word'].iloc[5])

    x = np.arange(len(Articles))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, positive, width, label='Positive Words')
    rects2 = ax.bar(x + width / 2, negative, width, label='Negative Words')

    ax.set_ylabel('Words Count')
    ax.set_title('Number of Word Count for 5 countries')
    ax.set_xticks(x, Articles)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()


# Problem 2
apikey = "Ur own api"
centerindex = 0


def generateCSV(criteria):
    df = pd.read_csv('Route/raw_data.csv')
    correct_state = df['state'] == criteria
    state_df = df[correct_state]
    arr = state_df.filter(items=['latitude', 'longitude']).values
    filename = "Route/" + criteria + ".csv"
    file = open(filename, 'w', encoding='UTF8', newline='')
    writer = csv.writer(file)
    for x in arr:
        writer.writerow(x)
    file.close


def choose7RandomBranches(state):
    filename = "Route/" + state + ".csv"
    df = pd.read_csv(filename, header=None)
    arr = np.array(df)
    arr2 = arr.tolist()
    randomIndex = random.sample(range(0, len(arr)), 7)
    coordinates = []
    for x in randomIndex:
        coordinates.append(arr2[x])
    return coordinates


def generatingPossibleRoute(branches, state, useolddata):
    filename = "Route/RouteCostFor_" + state + ".csv"
    if useolddata & exists(filename):
        # read the csv and write into AllInfo, then return
        # if use old data, random branches must be manually defined as same as in the csv
        AllInfo = []
        df = pd.read_csv(filename)
        arr = np.array(df)
        arr2 = arr.tolist()
        for i in arr2:
            AllInfo.append(i)
        return AllInfo
    list_of_jsonURL = []
    destination = ""
    for y in range(len(branches)):
        destination += str(branches[y])[1:len(str(branches[y])) - 1].replace(" ", "")
        if y == len(branches) - 1:
            break
        destination += ";"
    # print(destination)
    url1 = "https://dev.virtualearth.net/REST/v1/Routes/DistanceMatrix?origins="
    url2 = "&destinations="
    url3 = "&travelMode=driving&key="

    # Generate json URL
    for i in branches:
        temp2 = str(i)
        url = url1 + temp2[1:len(temp2) - 1].replace(" ", "") + url2 + destination + url3 + apikey
        print(url)
        list_of_jsonURL.append(url)
    # 1*7=7 pairs per http, total 7 loops hence 7 http, 7*7=49 pairs (including self-to-self because troublesome to remove and replace back)
    # Store json URL
    outputList = []
    for i in list_of_jsonURL:
        output = requests.get(i).json()
        outputList.append(output)

    # Extract destination Index and travel distance from origin to destination
    # Store the info in AllInfo list
    AllInfo = []
    filename = "Route/RouteCostFor_" + state + ".csv"
    file = open(filename, 'w', encoding='UTF8', newline='')
    writer = csv.writer(file)
    writer.writerow(["Origin", "location1", "destination", "location2", "distance"])
    for k in range(len(outputList)):
        for j in range(len(branches)):
            info = [
                k, str(branches[k])[1:len(str(branches[k])) - 1].replace(" ", ""),
                outputList[k]["resourceSets"][0]["resources"][0]["results"][j]["destinationIndex"],
                str(branches[j])[1:len(str(branches[j])) - 1].replace(" ", ""),
                outputList[k]["resourceSets"][0]["resources"][0]["results"][j]["travelDistance"]
            ]
            writer.writerow(info)
            AllInfo.append(info)
    file.close
    return AllInfo


def distributionCenter(branches, state, useolddata):
    # Choose the lowest std dev as distribution center
    allRoutes = generatingPossibleRoute(branches, state, useolddata)
    distances = []
    for i in range(len(allRoutes)):
        distances.append(allRoutes[i][4])
    stdList = []
    for i in range(0, len(allRoutes), 7):
        stdList.append(numpy.std(distances[i:i + 7]))
    global centerindex
    centerindex = stdList.index(min(stdList))
    return branches[centerindex]


# gn=destination to center
def getGN(df):
    gn = []
    df = df.loc[(df['destination'] == centerindex) & (df['distance'] != 0)]
    gn.append(df['distance'].values)
    return gn[0]


# hn=distance between 2 points
# [[0 to 1, 0 to 2,0 to 3 ...],[1 to 0, 1 to 2, 1 to 3...],...]
def getHN(df):
    hn = []
    tempHn = df["distance"].values
    listing = []
    for i in tempHn:
        if i != 0:
            listing.append(i)
    temp = []
    count = 0
    for i in range(len(listing)):
        if count % 5 == 0 and count != 0:
            temp.append(listing[i])
            hn.append(temp)
            count = 0
            temp = []
        else:
            temp.append(listing[i])
            count += 1

    return hn


RouteForPlotting = []
totalDistances = []


def A_star_search(fn, open_node, closed_node, branches, df):
    # fn is a 2d array which recorded the costs as [[0 to 1, 0 to 2,0 to 3 ...],[1 to 0, 1 to 2, 1 to 3...],...]
    if not bool(open_node):
        totalDistance = 0
        closed_node.append(centerindex)
        print("The route is: " + str(closed_node))
        coordinate = ""
        ctr = 0
        for i in range(len(closed_node)):
            coordinate += str(branches[closed_node[i]])
            RouteForPlotting.append(branches[closed_node[i]])
            if ctr < len(closed_node) - 1:
                totalDistance += df['distance'].where(
                    (df['Origin'] == closed_node[i]) & (df['destination'] == closed_node[i + 1])).sum()
                coordinate += "->"
                ctr += 1
        print(coordinate)
        totalDistances.append(totalDistance)
        print("Total Distance = " + str(totalDistance) + " km")
        viewLocation = plotPushPins(RouteForPlotting)
        print("Click here to see the center:")
        print(viewLocation)
        viewMap = plotRouteInBingMap(RouteForPlotting)
        print("Click here to see the map:")
        print(viewMap + "\n")
        return
    if not bool(closed_node):
        open_node.remove(centerindex)
        closed_node.append(centerindex)
    still_open = False
    current_node = closed_node[len(closed_node) - 1]
    while not still_open:
        min_index = fn[current_node].index(min(fn[current_node]))
        actual_index = fn[current_node].index(min(fn[current_node]))
        if actual_index >= current_node:
            min_index += 1
        for i in open_node:
            if i == min_index:
                still_open = True
                break
        if not still_open:
            fn[current_node][actual_index] = 999999
    open_node.remove(min_index)
    closed_node.append(min_index)
    A_star_search(fn, open_node, closed_node, branches, df)


def getFN(h, g):
    fn = []
    allFN = []
    for i in h:
        for x in range(6):
            allFN.append(i[x] + g[x])

    temp = []
    count = 0
    for i in range(len(allFN)):
        if count % 5 == 0 and count != 0:
            temp.append(allFN[i])
            fn.append(temp)
            count = 0
            temp = []
        else:
            temp.append(allFN[i])
            count += 1

    return fn


# Plot pushpins only to show the location of "CENTER DISTRIBUTION"
def plotPushPins(location):
    starter = "&dcl=1&mapSize=1000,750&key="
    api_key = starter + "Ur own api"
    URLstruture = "https://dev.virtualearth.net/REST/v1/Imagery/Map/Road?"
    URL = ""
    f1 = "&pp="
    f2 = ","
    f3 = ";;"
    f4 = "C"
    index = 0
    for i in location:
        if index == 0:
            URL = URL + f1 + str(i[0]) + f2 + str(i[1]) + f3 + f4
        else:
            URL = URL + f1 + str(i[0]) + f2 + str(i[1]) + f3 + str(index)
        index += 1
        if index == 8:
            break

    URL = URL + api_key
    link = URLstruture + URL
    return link


# To generete BING MAP showing the complete route from origin to different location
# Finally back to origin
def plotRouteInBingMap(RouteList):
    starter = "&dcl=1&mapSize=1000,750&optimize=distance&key="
    api_key = starter + "Ur own api"
    URLstructure = "https://dev.virtualearth.net/REST/v1/Imagery/Map/Road/Routes?"
    URL = ""
    f1 = "wp."
    f2 = ","
    index = 0
    f3 = ";64;"
    f4 = "="
    f5 = ";66;"
    f6 = "&"
    for x in RouteList:
        if index == 0:
            URL = URL + f1 + str(index) + f4 + str(x[0]) + f2 + str(x[1]) + f3 + str(index)
        else:
            URL = URL + f6 + f1 + str(index) + f4 + str(x[0]) + f2 + str(x[1]) + f5 + str(index)
        index += 1
        if index == 8:
            URL = URL + api_key
            break
    link = URLstructure + URL
    RouteForPlotting.clear()
    return link


# Problem 3

# Result from Problem 1 (Assumption only)
df_for_word = pd.read_csv('csv/Overview.csv')
good_words = df_for_word['positive_word'].tolist()
bad_words = df_for_word['negative_word'].tolist()

# Result from Problem 2
# distance_list = [356.658, 739.394, 70.543, 1326.8799, 10986.241] # Sample
distance_list = totalDistances


def wordRatio(goodwords, badwords):
    country = ["AE", "MY", "SG", "UK", "US"]
    problist = []
    for i in range(len(country)):
        ratio = (goodwords[i]) / (goodwords[i] + badwords[i])
        problist.append(ratio)
    return problist


def distanceRatio():
    country = ["AE", "MY", "SG", "UK", "US"]
    distance_ratio = []
    for i in range(len(country)):
        distance_ratio.append(1 / (distance_list[i] / sum(distance_list)))
    normalized = preprocessing.normalize([distance_ratio])
    return normalized[0]


def calculateScore(score1, score2):
    country = ["AE", "MY", "SG", "UK", "US"]
    score = []
    for i in range(len(country)):
        score.append(round(((score1[i] * score2[i]) * 100), 5))
    return score


# -----------------------------------Runner code---------------------------------------------#
if __name__ == "__main__":
    # Problem 1
    while True:
        filename = input("\nEnter filename (or type '...' to exit): ").upper()
        if filename == "...":
            break
        file = "Text/" + filename + ".txt"

        try:
            with open(file, "r", encoding="utf8") as f:
                content = f.read().lower().split()
                content = filter_stopwords(content)
                good_arr, bad_arr, neutral_arr = sort_words(content)

                good_arr = word_freq(good_arr)
                bad_arr = word_freq(bad_arr)
                neutral_arr = word_freq(neutral_arr)

                good_df = pd.DataFrame(good_arr, columns=['Word', 'Frequency'])
                bad_df = pd.DataFrame(bad_arr, columns=['Word', 'Frequency'])

                good_df.to_csv("csv/good-" + filename + ".csv", index=False)
                bad_df.to_csv("csv/bad-" + filename + ".csv", index=False)

                tot_good = count_words(good_arr)
                tot_bad = count_words(bad_arr)
                tot_neut = count_words(neutral_arr)

                csv_file = "csv/" + filename[:2] + ".csv"
                overview_file = "csv/Overview.csv"
                data = {
                    'article': [filename],
                    'positive_words': [tot_good],
                    'negative_words': [tot_bad]
                }
                df = pd.DataFrame(data)
                df.to_csv(csv_file, mode='a', index=False, header=False)
                df.to_csv(overview_file, mode='a', index=False, header=False)

                print("File: %s\nTotal positive words: %d\nTotal negative words: %d\nTotal neutral words: %d\n"
                      % (filename, tot_good, tot_bad, tot_neut))

        except IOError as e:
            print("Error{0}: {1}".format(e.errno, e.strerror))

    YesOrNo = int(input("Enter your 1 for plotting or 0 to continue the program: "))
    if YesOrNo == 1:
        countries = ["AE", "MY", "SG", "UK", "US"]
        for i in countries:
            plotChart(i)

        # Plot overview
        plotOverview()
    choice = int(input("Enter 1 to continue with problem 2: "))

    # Problem 2
    print("\nProblem 2")
    states_p2 = ["AE", "MY", "SG", "GB", "US"]
    for x in states_p2:
        generateCSV(x)
        randomBranches = choose7RandomBranches(x)
        print(x + "'s Randomly Selected Branches: ")
        print(randomBranches)
        center = distributionCenter(randomBranches, x, False)
        # If you want to use old data without api calls, change to true
        # For without api call, if the csv file does not exist, it will still proceed with api call and generate file
        print("Distribution center for " + x + " is " + str(center))
        print("Index of distribution center: " + str(centerindex))
        df = pd.read_csv('Route/RouteCostFor_' + x + '.csv')
        h = getHN(df)
        g = getGN(df)
        # f(n) = g(n) + h(n)
        f = getFN(h, g)
        A_star_search(f, [0, 1, 2, 3, 4, 5, 6], [], randomBranches, df)

    choice = int(input("Enter 1 to continue problem 3: "))

    # Problem 3
    country_p3 = ["AE", "MY", "SG", "GB", "US"]
    print("\nProblem 3")
    scoreA = wordRatio(good_words, bad_words)
    print("Word Score For Each Country:")
    for i in range(len(country_p3)):
        print(str(country_p3[i]) + ": " + str(scoreA[i]))

    scoreB = distanceRatio()
    print("\nDistance Score For Each Country:")
    for i in range(len(country_p3)):
        print(str(country_p3[i]) + ": " + str(scoreB[i]))

    finalScore = calculateScore(scoreA, scoreB)
    print("\nScore for all country based on the local economic and lowest optimal delivery:")
    for i in range(len(country_p3)):
        print(str(country_p3[i]) + ": " + str(finalScore[i]) + " %")

    print("\nThe most recommended countries based on optimal distance and positive sentiment is",
          country_p3[finalScore.index(max(finalScore))] + ".")

    choice = int(input("\nEnter your 0 to exit: "))
    SystemExit
