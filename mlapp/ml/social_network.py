import logging
import os

import matplotlib.pyplot as plt
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI environments

import matplotlib.pyplot as plt
import os


class DynamicVirtualUser:
    def __init__(self, user_id, susceptibility, interests):
        self.id = user_id
        self.susceptibility = susceptibility
        self.interests = interests
        self.friends = []
        self.read_articles = []

    def post_article(self, article_id, current_time, network):
        """Publikuje artykuł i informuje znajomych."""
        logging.info(f"User {self.id} posted article {article_id} at time {current_time}")
        for friend_id in self.friends:
            friend = network.nodes[friend_id]['user']
            if not any(article_id == read_article_id for read_article_id, _ in friend.read_articles):
                friend.read_article(article_id, current_time + 1)
                logging.info(f"User {friend_id} received and read article {article_id} at time {current_time + 1}")

    def read_article(self, article_id, read_time):
        """Oznacza artykuł jako przeczytany przez użytkownika."""
        self.read_articles.append((article_id, read_time))
        logging.info(f"User {self.id} read article {article_id} at time {read_time}")

def create_dynamic_virtual_users(num_users, interests):
    """Tworzy dynamicznych użytkowników z losowymi zainteresowaniami i podatnością na fake newsy."""
    dynamic_users = []
    for i in range(num_users):
        susceptibility = random.uniform(0, 1)
        user_interests = random.sample(interests, k=random.randint(1, len(interests)))
        dynamic_users.append(DynamicVirtualUser(i, susceptibility, user_interests))
    return dynamic_users

def create_social_network(users, average_friends=5):
    """Tworzy sieć społecznościową, łącząc użytkowników w sieć znajomych."""
    G = nx.Graph()
    for user in users:
        G.add_node(user.id, user=user)
        friends = random.sample([u.id for u in users if u.id != user.id], k=min(average_friends, len(users)-1))
        for friend in friends:
            G.add_edge(user.id, friend)
            user.friends.append(friend)
    return G


import random


def simulate_information_spread(dynamic_users, social_network, articles):
    simulation_results = {
        'user_actions': {user_id: [] for user_id in dynamic_users},
        'daily_infections': [],
        'cumulative_infections': []
    }

    cumulative_infected = 0

    for day in range(1, 31):  # Simulate over 30 days
        daily_infected = 0

        for user_id in dynamic_users:
            for article_id in articles['id']:
                # Simulate a user reading an article
                simulation_results['user_actions'][user_id].append((article_id, 'read'))

                # Simulate a probability that the user will share the article
                if random.random() < 0.2:  # Example probability
                    simulation_results['user_actions'][user_id].append((article_id, 'share'))
                    daily_infected += 1

        # Update daily and cumulative infections
        simulation_results['daily_infections'].append(daily_infected)
        cumulative_infected += daily_infected
        simulation_results['cumulative_infections'].append(cumulative_infected)

    return simulation_results

def plot_simulation_results(simulation_results, num_users, save_path):
    # Accessing the 'user_actions' dictionary
    user_actions = simulation_results['user_actions']

    num_reads = []
    num_shares = []

    # Iterate over the user actions
    for user, actions in user_actions.items():
        reads = sum(1 for _, action_type in actions if action_type == 'read')
        shares = sum(1 for _, action_type in actions if action_type == 'share')

        num_reads.append(reads)
        num_shares.append(shares)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.bar(['Reads', 'Shares'], [sum(num_reads), sum(num_shares)])
    plt.title('Total Reads vs Shares')
    plt.savefig(f'{save_path}/simulation_results_summary.png')
    plt.close()

def plot_article_read_share_frequency(simulation_results, save_path):
    # Initialize dictionaries to hold read and share counts per article
    read_counts = {}
    share_counts = {}

    # Extract user actions from the simulation results
    user_actions = simulation_results['user_actions']

    # Iterate through the actions for each user
    for user, actions in user_actions.items():
        for action in actions:
            article_id, action_type = action

            if action_type == 'read':
                if article_id in read_counts:
                    read_counts[article_id] += 1
                else:
                    read_counts[article_id] = 1
            elif action_type == 'share':
                if article_id in share_counts:
                    share_counts[article_id] += 1
                else:
                    share_counts[article_id] = 1

    # Convert to lists for plotting
    articles = list(read_counts.keys())
    reads = [read_counts.get(article, 0) for article in articles]
    shares = [share_counts.get(article, 0) for article in articles]

    # Plotting the results
    plt.figure(figsize=(14, 7))
    plt.bar(articles, reads, label='Reads', alpha=0.7)
    plt.bar(articles, shares, label='Shares', alpha=0.7, bottom=reads)
    plt.xlabel('Article ID')
    plt.ylabel('Count')
    plt.title('Article Read and Share Frequency')
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'{save_path}/article_read_share_frequency.png')
    plt.close()

def plot_daily_infections(simulation_results, save_path):
    daily_infections = simulation_results.get('daily_infections', [])

    if daily_infections:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(daily_infections) + 1), daily_infections, marker='o')  # Use marker 'o'
        plt.title('Daily New Infections')
        plt.xlabel('Day')
        plt.ylabel('Number of New Infections')
        plt.savefig(os.path.join(save_path, 'daily_infections.png'))
        plt.close()
    else:
        print("No data available for daily infections.")

def plot_cumulative_infections(simulation_results, save_path):
    cumulative_infections = simulation_results.get('cumulative_infections', [])

    if cumulative_infections:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_infections) + 1), cumulative_infections, marker='o')  # Use marker 'o'
        plt.title('Cumulative Infections Over Time')
        plt.xlabel('Day')
        plt.ylabel('Total Number of Infections')
        plt.savefig(os.path.join(save_path, 'cumulative_infections.png'))
        plt.close()
    else:
        print("No data available for cumulative infections.")
