class evaluation():
    def __init__(self, name):
        self.filename = name
        self.count = 0

    def eval_sst2(self,gt,pred):
        with open(self.filename, "w", encoding="utf-8") as file:
            for i in range(len(pred)):
                if (pred[i] == "\"Pos" or pred[i] == "Positive\n" or pred[i] == "Class1" or pred[i] == "positive\n" or pred[i] == "positive ") and gt[i] == 1:
                    pass
                elif (pred[i] == "\"Neg" or pred[i] == "Negative\n" or pred[i] == "Class0" or pred[i] == "negative\n" or pred[i] == "negative ") and gt[i] == 0:
                    pass
                else:
                    print(i, '-', gt[i], '-', pred[i])
                    self.count += 1
                    file.write("{}-{}-{}\n".format(i, gt[i], pred[i]))
            file.write(f"{self.count}")

        print(self.count)

    def eval_cola(self,gt,pred):

        with open(self.filename, "w",encoding="utf-8") as file:
            for i in range(len(pred)):
                if (pred[i] == "Class1" or pred[i] == "acceptabl"
                    or pred[i] == "acceptable" or pred[i] == "\"acceptable" or pred[i] == "acceptable ") and gt[i] == 1:
                    pass
                elif (pred[i] == "Class0" or pred[i] == "\"unacceptable\""
                      or pred[i] == "unacceptable\n" or pred[i] == "unaccept" or pred[i] == "unacceptable") and gt[i] == 0:
                    pass
                else:
                    print(i, '-', gt[i], '-', pred[i])
                    file.write("{}-{}-{}\n".format(i, gt[i], pred[i]))
                    self.count += 1
            file.write(f"{self.count}")

        print(self.count)

    def eval_emotion(self,gt,pred):
        #    0: 'sadness',
        #    1: 'joy',
        #    2: 'love',
        #    3: 'anger',
        #    4: 'fear',
        #    5: 'surprise'
        with open(self.filename, "w", encoding="utf-8") as file:
            count = 0
            for i in range(len(pred)):
                if (pred[i] == 'joy\n' or pred[i] == 'joy') and gt[i] == 1:
                    pass
                elif (pred[i] == 'sadness\n' or pred[i] == 'sadness') and gt[i] == 0:
                    pass
                elif (pred[i] == 'love\n' or pred[i] == 'love') and gt[i] == 2:
                    pass
                elif (pred[i] == 'anger\n' or pred[i] == 'anger') and gt[i] == 3:
                    pass
                elif (pred[i] == 'fear\n' or pred[i] == 'fear') and gt[i] == 4:
                    pass
                elif (pred[i] == 'surprise\n' or pred[i] == 'surprise') and gt[i] == 5:
                    pass
                else:
                    print(i, '-', gt[i], '-', pred[i])
                    file.write("{}-{}-{}\n".format(i, gt[i], pred[i]))
                    count += 1
            file.write(f"{self.count}")
            print(count)

    def eval_bbc(self,gt,pred):
        # 0: tech 1: business 2: sport 3: entertainment 4: politics
        with open(self.filename, "w", encoding="utf-8") as file:
            count = 0
            for i in range(len(pred)):
                if (pred[i] == 'tech\n' or pred[i] == 'tech \n') and gt[i] == 0:
                    pass
                elif (pred[i] == 'business\n' or pred[i] == 'business \n') and gt[i] == 1:
                    pass
                elif (pred[i] == 'sport\n' or pred[i] == 'sport \n') and gt[i] == 2:
                    pass
                elif (pred[i] == 'entertainment\n' or pred[i] == 'entertainment \n') and gt[i] == 3:
                    pass
                elif (pred[i] == 'politics\n' or pred[i] == 'politics \n') and gt[i] == 4:
                    pass
                else:
                    print(i, '-', gt[i], '-', pred[i])
                    file.write("{}-{}-{}\n".format(i, gt[i], pred[i]))
                    count += 1
            file.write(f"{count}")
            print(count)