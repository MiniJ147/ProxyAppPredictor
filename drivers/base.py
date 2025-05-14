import time
import math
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import os
import matplotlib.pyplot as plt


class Base:
    def setup_figures(self,model_name):
        try:
            os.makedirs("figures/"+str(model_name).replace(" ","_"))
        except FileExistsError:
            # Directory already exists
            pass


    def run(self,regressor,model_name,X,y,*args):
        """
        this will run the regressor and return the y_pred and results
        """
        print(model_name,"running base driver")
        self.setup_figures(model_name)
        ret = str(model_name) + "\n"

        # DEBUG
        y_pred = []
        # Some predictors only work properly when given one job at a time.
        startTime = time.process_time()
        regressor = regressor.fit(X, y)
        endTime = time.process_time()
        total_time = endTime - startTime
    
        print(model_name,"trained",total_time)

        # Retrain on 4/5 of the data for plotting.
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
        plot_regressor = regressor.fit(X_train, y_train)

        scores = cross_val_score(regressor, X, y, cv=5,
                                 scoring="r2")

        y_pred = plot_regressor.predict(X_test)

                    # Report cross-validation accuracy.
        ret += " R^2: " + str(scores.mean()) + "\n"
        ret += str(total_time) + "s \n"

        plt.scatter(y_pred, y_test, s=20, c="black", label="data")

        plt.xlabel("Predicted (minutes) ("+str(model_name)+")")
        plt.ylabel("Actual (minutes)")
        try:
            os.makedirs("figures")
        except FileExistsError:
            # Directory already exists
            pass
        plt.savefig("figures/"+str(model_name).replace(" ", "_")+".svg")
        plt.close()


        return ret 
'''
[TODO]: 2 models (train then predict), (predict/train) Complete, Trasfiri
page 6 of paper Random Forest - 2 PLS (train -> predict), others don't have a specfic training step
can kinda of look at the one_at_a_time flag in the regession function, but it does get passed around (maybe..)


testing/training, creating new jobs, running jobs
'''
class Single(Base):
    def run(self,regressor,model_name,X,y,*args):
        super().setup_figures(model_name)
        assert len(X) == len(y)
        y_pred = []
        for i in range(len(X)):
            xVal = X.iloc[i]
            yVal = y.iloc[i]
            prediction = regressor.predict_and_fit(xVal, yVal)
            y_pred.append(prediction)


        # R^2 calculation since cross_val_score isn't meaningful here.
        numerator = ((y - y_pred) ** 2).sum(axis=0)
        denominator = ((y - np.average(y, axis=0)) ** 2).sum(axis=0)
        score = 1 - numerator / denominator
        ret = model_name+" R^2: " + str(score) + "\n"
        plt.scatter(y_pred, y, s=20, c="black", label="data")
        plt.savefig("figures/"+str(model_name).replace(" ","_")+".svg")

        return ret
        


class Quantile(Base):
    def run(self,regressor,model_name,X,y,*args):
        super().setup_figures(model_name)
        ret = str(model_name) + "\n"
        assert len(args)>0, "need quantiles arg"
        quantiles = args[0]

        # DEBUG
        y_pred = []
        # Some predictors only work properly when given one job at a time.
        startTime = time.process_time()
        regressor = regressor.fit(X, y)
        endTime = time.process_time()
        total_time = endTime - startTime
        # Retrain on 4/5 of the data for plotting.
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
        plot_regressor = regressor.fit(X_train, y_train)

        scores = cross_val_score(regressor, X, y, cv=5,
                                 scoring="r2")

        y_pred = plot_regressor.predict(X_test,quantiles=quantiles)

                    # Report cross-validation accuracy.
        ret += " R^2: " + str(scores.mean()) + "\n"
        ret += str(total_time) + "s \n"

        BOUND = len(y_test)
        samples = len(quantiles)
        confidence_lvl = quantiles[-1]-quantiles[0]

                        # doing data analysis
        for i in range(samples):
            # _, ax = plt.subplots()
            quart = round(quantiles[i] * 100,2)
            guesses = y_pred[:,i]
            over_est = under_est = prefect_est = 0
                
                
            MAX_FAIL_PRECT = 200.0
            fail = 0
            domain = []
            for j in range(len(y_test)):
                guess = guesses[j]
                real = y_test.values[j]
                if real == 0: continue

                div = int((guess/real)*100)

                if div > MAX_FAIL_PRECT:
                    fail+=1

                if guess > real:
                    over_est += 1
                elif guess < real:
                    under_est += 1
                else:
                    prefect_est += 1

                domain.append(div) 

            print(f"{quart}th\nOver Est: {over_est}\n Under Est: {under_est}\n Prefect Est: {prefect_est}\nFails: {fail}")

            #fixed with
            lower = int(min(domain))
            upper = int(math.ceil(max(domain)))
            bins = [0,60,80,90,100,140,180,200,max(201,upper)]

            counts,edges = np.histogram(domain,bins=bins)
            size = len(edges); cnt_sum = counts.sum()
            vals = []; cats = []


            # summing for our labels
            skips = [[90,0],[100,0],[200,0]]
            sidx = 0
            for i in range(size-1):
                if edges[i] >= skips[sidx][0]:
                    sidx+=1
                prob = round(counts[i]/cnt_sum,5)
                if i+1 == size-1: 
                    cats.append(">=200%")
                    skips.append([201,prob])
                else: 
                    cats.append(f"[{edges[i]}%,{edges[i+1]}%)")
                    skips[sidx][1]+=prob
                vals.append(prob)

            key = f"Probability:\n<90% = {skips[0][1]:.5f}%\n[90%,100%): = {skips[1][1]:.5f}%\n[100%,200%): = {skips[2][1]:.5f}%\n>200% = {skips[3][1]:.5f}%"
                
            plt.xlabel(f"Precentage of Real Runtime")
            plt.ylabel("Probability")
            plt.title(f"{model_name}-{quart}th-distribution")


            plt.annotate(key,xy=(0,0),xytext=(0,-40),xycoords='axes fraction',textcoords='offset points',va='top',ha='left')
            plt.bar(cats,vals)
            plt.tick_params(axis='x',rotation=25,which="major",labelsize=8)
            plt.tight_layout()

            fig_name = str(model_name).replace(" ","_")
            plt.savefig(f"figures/{fig_name}/{quart}th_distrubtion.svg")
            plt.close()
        return ret
