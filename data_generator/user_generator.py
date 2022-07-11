import numpy as np
import sys
import pickle


class User:
    MAX_LOW = 0.1
    MIN_HIGH = 0.65

    def __init__(self, topic_preferences=[],
                 topic_knowledge=[],
                 critical_thinking=True,
                 fake_preference=False,
                 viral_preference=False,
                 recognize_manipulation=False):
        self.topic_preferences = topic_preferences
        self.topic_knowledge = topic_knowledge
        self.critical_thinking = critical_thinking
        self.fake_preference = fake_preference
        self.viral_preference = viral_preference
        self.recognize_manipulation = recognize_manipulation
        self.user_feature_vector = np.concatenate((self.init_topic_preference(*self.topic_preferences),
                                                   self.init_topic_knowledge(*self.topic_knowledge),
                                                   self.init_critical_thinking(self.critical_thinking),
                                                   self.init_fake_preference(self.fake_preference),
                                                   self.init_viral_preference(self.viral_preference),
                                                   self.init_recognize_manipulation(self.recognize_manipulation)))

    def init_topic_preference(self, *topics):
        if len(topics) > 4:
            sys.exit(0)
        topic_vector = np.random.uniform(low=0, high=self.MAX_LOW, size=4)
        for topic in topics:
            if topic == 'space':
                topic_vector[0] = np.random.uniform(low=self.MIN_HIGH, high=1)
            elif topic == 'politics':
                topic_vector[1] = np.random.uniform(low=self.MIN_HIGH, high=1)
            elif topic == 'naturaldisaster':
                topic_vector[2] = np.random.uniform(low=self.MIN_HIGH, high=1)
            elif topic == 'commons':
                topic_vector[3] = np.random.uniform(low=self.MIN_HIGH, high=1)
            else:
                sys.exit(0)
        return topic_vector

    def init_topic_knowledge(self, *topics):
        if len(topics) > 4:
            sys.exit(0)
        topic_vector = np.random.uniform(low=0, high=self.MAX_LOW, size=4)
        for topic in topics:
            if topic == 'space':
                topic_vector[0] = np.random.uniform(low=self.MIN_HIGH, high=1)
            elif topic == 'politics':
                topic_vector[1] = np.random.uniform(low=self.MIN_HIGH, high=1)
            elif topic == 'naturaldisaster':
                topic_vector[2] = np.random.uniform(low=self.MIN_HIGH, high=1)
            elif topic == 'commons':
                topic_vector[3] = np.random.uniform(low=self.MIN_HIGH, high=1)
            else:
                sys.exit(0)
        return topic_vector

    def init_critical_thinking(self, critical_thinking):
        if critical_thinking:
            return np.random.uniform(low=self.MIN_HIGH, high=1, size=1)
        else:
            return np.random.uniform(low=0, high=self.MAX_LOW, size=1)

    def init_fake_preference(self, fake_preference):
        if fake_preference:
            return np.random.uniform(low=self.MIN_HIGH, high=1, size=1)
        else:
            return np.random.uniform(low=0, high=self.MAX_LOW, size=1)

    def init_viral_preference(self, viral_preference):
        if viral_preference:
            return np.random.uniform(low=self.MIN_HIGH, high=1, size=1)
        else:
            return np.random.uniform(low=0, high=self.MAX_LOW, size=1)

    def init_recognize_manipulation(self, recognize_manipulation):
        if recognize_manipulation:
            return np.random.uniform(low=self.MIN_HIGH, high=1, size=1)
        else:
            return np.random.uniform(low=0, high=self.MAX_LOW, size=1)

    def get_user_features(self):
        return self.user_feature_vector


def save_user(user, filename: str):
    with open(filename+'.pickle', 'wb') as handle:
        pickle.dump({'user': user}, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_user(filename: str):
    with open(filename+'.pickle', 'rb') as handle:
        return pickle.load(handle)['user']


def generate_random_user():
    ntopics = np.random.choice([0,1,2,3,4])
    nknowledge = np.random.choice([0,1,2,3,4])
    topics = np.random.choice(['space', 'commons', 'politics', 'naturaldisaster'], size= ntopics,replace=False)
    knowledge = np.random.choice(['space', 'commons', 'politics', 'naturaldisaster'], size= nknowledge, replace=False)
    t_f = np.random.choice([True, False], size=4, replace=True)
    test_user = User(topic_preferences=topics,
                     topic_knowledge=knowledge,
                     critical_thinking=t_f[0],
                     fake_preference=t_f[1],
                     viral_preference=t_f[2],
                     recognize_manipulation=t_f[3])
    return test_user


if __name__ == '__main__':
    test_user = User(topic_preferences=['space', 'politics', 'naturaldisaster'],
                     topic_knowledge=['commons'],
                     critical_thinking=False, 
                     fake_preference=True, 
                     viral_preference=True,
                     recognize_manipulation=True)
    print(test_user.get_user_features())
