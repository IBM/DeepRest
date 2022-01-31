from locust import LoadTestShape, HttpUser, task, between, events
import numpy as np
import resource
import pickle
import random
import math
import os
resource.setrlimit(resource.RLIMIT_NOFILE, (250000, 250000))


####################################################################################################################################
# Simulation Configuration
####################################################################################################################################
GLOBAL_NGINX_FRONTEND_URL  = 'CHANGE_THIS_URL'
GLOBAL_MEDIA_FRONTEND_URL  = 'CHANGE_THIS_URL'

GLOBAL_EXPERIMENT_DURATION = 3600    # None = Run forever, 43200 = 12 hour
GLOBAL_SECONDS_PER_DAY     = 3600    # 3600 = 1 hour
GLOBAL_MIN_USERS           = 100
GLOBAL_PEAKS               = [140, 160, 180, 200]
GLOBAL_RANDOMNESS          = 0.20
GLOBAL_WAIT_TIME           = between(1, 3)
GLOBAL_COMPOSITIONS        = [(5, 45, 50), (10, 40, 50), (15, 35, 50), (10, 45, 45), (10, 50, 40), (5, 55, 40), (15, 40, 45), (10, 35, 55), (10, 55, 35), (5, 40, 55), (15, 45, 40), (5, 50, 45), (15, 50, 35)]


####################################################################################################################################
texts = [text.replace('@', '') for text in list(open('./datasets/fb-posts/news.txt'))]
media = [os.path.join('./datasets/inria-person', fname) for fname in os.listdir('./datasets/inria-person')]
users = list(range(1, 963))
cycle = 0
active_users, inactive_users = [], list(range(1, 963))
with open('./datasets/social-graph/socfb-Reed98.mtx', 'r') as f:
    friends = {}
    for edge in f.readlines():
        edge = list(map(int, edge.strip().split()))
        if len(edge) == 0:
            continue
        if edge[0] not in friends:
            friends[edge[0]] = set()
        if edge[1] not in friends:
            friends[edge[1]] = set()
        friends[edge[0]].add(edge[1])
        friends[edge[1]].add(edge[0])
    friends = {user: list(l) for user, l in friends.items()}

####################################################################################################################################
class LoadShape(LoadTestShape):
    peak_one_users = None
    peak_two_users = None
    second_of_day = None


    def tick(self):
        global cycle
        if GLOBAL_EXPERIMENT_DURATION is not None and round(self.get_run_time()) > GLOBAL_EXPERIMENT_DURATION:
            return None

        second_of_day = round(self.get_run_time()) % GLOBAL_SECONDS_PER_DAY
        if self.second_of_day is None or second_of_day < self.second_of_day:
            cycle += 1
            self.peak_one_users = random.choice(GLOBAL_PEAKS)
            self.peak_two_users = random.choice(GLOBAL_PEAKS)
        self.second_of_day = second_of_day

        user_count = max(self.peak_one_users, self.peak_two_users)
        max_offset = math.ceil(user_count * GLOBAL_RANDOMNESS)
        user_count += random.choice(list(range(-max_offset, max_offset + 1)))
        return round(user_count), round(min(user_count, 70))


class SocialNetworkUser(HttpUser):
    wait_time = GLOBAL_WAIT_TIME
    host = GLOBAL_NGINX_FRONTEND_URL
    local_cycle = cycle

    def check_cycle(self):
        if self.local_cycle != cycle:
            self.local_cycle = cycle
            composition = GLOBAL_COMPOSITIONS[self.local_cycle % len(GLOBAL_COMPOSITIONS)]
            self.tasks = [self.apis[0] for _ in range(composition[0])] + [self.apis[1] for _ in range(composition[1])] + [self.apis[2] for _ in range(composition[2])]

    @task
    def composePost(self):
        self.check_cycle()

        text = random.choice(texts)

        # User mentions
        number_of_user_mentions = random.randint(0, min(5, len(friends[self.user_id])))
        if number_of_user_mentions > 0:
            for friend_id in random.choices(friends[self.user_id], k=number_of_user_mentions):
                text += " @username_" + str(friend_id)
        # Media
        media_id = ''
        media_type = ''
        if random.random() < 0.20:
            with open(random.choice(media), "rb") as f:
                media_response = self.client.post('%s/upload-media' % GLOBAL_MEDIA_FRONTEND_URL,
                                                  files={"media": f})
            if media_response.ok:
                media_json = eval(media_response.text)
                media_id = '"%s"' % media_json['media_id']
                media_type = '"%s"' % media_json['media_type']
        # URLs - Note: no need to add it as the original post content has URLs already

        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        data = {'username': 'username_' + str(self.user_id),
                'user_id': str(self.user_id),
                'text': text,
                'media_ids': "[" + str(media_id) + "]",
                'media_types': "[" + str(media_type) + "]",
                'post_type': '0'}

        self.client.post("/wrk2-api/post/compose", data=data, headers=headers)


    @task
    def readHomeTimeline(self):
        self.check_cycle()

        start = random.randint(0, 100)
        stop = start + 10

        response = self.client.get(
            "/wrk2-api/home-timeline/read?start=%s&stop=%s&user_id=%s" % (str(start), str(stop), str(self.user_id)),
            name="/wrk2-api/home-timeline/read?start=[start]&stop=[stop]")

    @task
    def readUserTimeline(self):
        self.check_cycle()

        start = random.randint(0, 100)
        stop = start + 10
        user_id = random.choice(friends[self.user_id])

        response = self.client.get(
            "/wrk2-api/user-timeline/read?start=%s&stop=%s&user_id=%s" % (str(start), str(stop), str(user_id)),
            name='/wrk2-api/user-timeline/read?start=[start]&stop=[stop]&user_id=[user_id]')

    apis = [composePost, readHomeTimeline, readUserTimeline]

    def on_stop(self):
        active_users.remove(self.user_id)
        inactive_users.append(self.user_id)

    def on_start(self):
        self.user_id = random.choice(inactive_users)
        active_users.append(self.user_id)
        inactive_users.remove(self.user_id)
