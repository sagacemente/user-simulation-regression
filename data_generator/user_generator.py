import numpy as np
import sys


class User:
    MAX_LOW = 0.1
    MIN_HIGH = 0.65

    def __init__(self, topic_preferences=[],
                 topic_knowledge=[],
                 critical_thinking=True,
                 fake_preference=False,
                 viral_preference=False):
        self.topic_preference_vec = self.init_topic_preference(*topic_preferences)
        self.topic_knowledge_vec = self.init_topic_knowledge(*topic_knowledge)
        self.critical_thinking = self.init_critical_thinking(critical_thinking)
        self.fake_preference = self.init_fake_preference(fake_preference)
        self.viral_preference = self.init_viral_preference(viral_preference)

    def init_topic_preference(self, *topics):
        if len(topics) > 4:
            sys.exit(0)
        topic_vector = np.random.uniform(low=0, high=self.MAX_LOW, size=4)
        for topic in topics:
            if topic == 'aerospace':
                topic_vector[0] = np.random.uniform(low=self.MIN_HIGH, high=1)
            elif topic == 'politics':
                topic_vector[1] = np.random.uniform(low=self.MIN_HIGH, high=1)
            elif topic == 'natural disaster':
                topic_vector[2] = np.random.uniform(low=self.MIN_HIGH, high=1)
            elif topic == 'common':
                topic_vector[3] = np.random.uniform(low=self.MIN_HIGH, high=1)
            else:
                sys.exit(0)
        return topic_vector

    def init_topic_knowledge(self, *topics):
        if len(topics) > 4:
            sys.exit(0)
        topic_vector = np.random.uniform(low=0, high=self.MAX_LOW, size=4)
        for topic in topics:
            if topic == 'aerospace':
                topic_vector[0] = np.random.uniform(low=self.MIN_HIGH, high=1)
            elif topic == 'politics':
                topic_vector[1] = np.random.uniform(low=self.MIN_HIGH, high=1)
            elif topic == 'natural disaster':
                topic_vector[2] = np.random.uniform(low=self.MIN_HIGH, high=1)
            elif topic == 'common':
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

    def get_user_features(self):
        return np.concatenate((self.topic_preference_vec,
                               self.topic_knowledge_vec,
                               self.critical_thinking,
                               self.fake_preference,
                               self.viral_preference))


if __name__ == '__main__':
    test_user = User(topic_preferences=['aerospace', 'politics'], topic_knowledge=['common'])
    print(test_user.get_user_features())
