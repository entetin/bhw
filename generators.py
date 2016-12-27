import pprint
import sys
from random import shuffle

def make_intent(example):       # returns a set of items "attribute:value"
    global attrib_names
    return set([i+":"+str(k) for i,k in zip(attrib_names,example)])

def make_closure(context, example_intent):  # makes it a tiny bit faster
    closure = []
    for i in context:
        mi = make_intent(i)
        if mi.issuperset(example_intent):
            closure.append(mi)
    return closure

def check_hypothesis(context_plus, context_minus, example, mode):
    eintent = make_intent(example)          # example intent
    eintent.discard("class:positive")
    eintent.discard("class:negative")
    eps = 0.001                             # difference threshold
    global cv_res
    labels = {}                             # a dict that stores weights for positive and negative results

    if mode == 0:
        for e in context_plus:                  # for every item in plus-context
            ei = make_intent(e)                 # make its intent
            candidate_intent = ei & eintent     # intersection of example and plus-item (their intents)
            # for each minus-item, if it contains that intersection, add the minus-item to the closure list
            closure = make_closure(context_minus, candidate_intent)
            closure_size = len([i for i in closure if len(i)])      # determine the size of the list
            # command below, possibly, combines "class:positive" and "class:negative" results
            res = reduce(lambda x,y: x&y if x&y else x|y, closure, set())
            for cs in ["positive","negative"]:
                if "class:"+cs in res:
                    labels[cs+"_total_weight"] = labels.get(cs+"_total_weight", 0) + closure_size*1.0/(len(context_minus)+len(context_plus))
        for e in context_minus:                 # the same loop for minus-context
            ei = make_intent(e)
            candidate_intent = ei & eintent
            closure = make_closure(context_plus, candidate_intent)
            closure_size = len([i for i in closure if len(i)])
            res = reduce(lambda x,y: x&y if x&y else x|y, closure, set())
            for cs in ["positive","negative"]:
                if "class:"+cs in res:
                    labels[cs+"_total_weight"] = labels.get(cs+"_total_weight", 0) + closure_size*1.0/(len(context_plus)+len(context_minus))
        pw = labels.get("positive_total_weight", 0)
        nw = labels.get("negative_total_weight", 0)
        if abs(pw - nw) < eps:
           cv_res["contradictory"] += 1
        elif example[-1] == "positive":
            if pw > nw:
                cv_res["true_positive"]  += 1
            else:
                cv_res["false_positive"] += 1
        elif example[-1] == "negative":
            if nw > pw:
                cv_res["true_negative"]  += 1
            else:
                cv_res["false_negative"] += 1

    elif mode == 1:
        prob_plus = len(context_plus)*1.0/(len(context_plus) + len(context_minus))
        prob_minus = 1 - prob_plus
        for i in eintent:
            feature = set([i])
            common_plus  = [feature & make_intent(x) for x in context_plus] 
            common_minus = [feature & make_intent(y) for y in context_minus]
            num_common_plus  = len([z for z in common_plus  if len(z) > 0])
            num_common_minus = len([p for p in common_minus if len(p) > 0])
            prob_plus  = prob_plus *num_common_plus /len(context_plus)
            prob_minus = prob_minus*num_common_minus/len(context_minus)

        if prob_plus > prob_minus:
            labels['positive'] = True
        elif prob_plus < prob_minus:
            labels['negative'] = True
        else:
            labels['contradictory'] = True

        if labels.get("contradictory", False):
            cv_res["contradictory"] += 1
        elif example[-1] == "positive":
            if labels.get("positive", False):
                cv_res["true_positive"]  += 1
            if labels.get("negative", False):
                cv_res["false_negative"] += 1
        elif example[-1] == "negative":
            if labels.get("positive", False):
                cv_res["false_positive"] += 1
            if labels.get("negative", False):
                cv_res["true_negative"]  += 1

def k_fold_cross_validation(X, K, randomize = False):
    if randomize:
        X = list(X)
        shuffle(X)
    for k in range(K):
        training   = [x for i, x in enumerate(X) if i % K != k]
        validation = [x for i, x in enumerate(X) if i % K == k]
        yield training, validation#, k+1

def metrics(results):
    m = {}
    acc  = div_by_zero(results["true_positive"] + results["true_negative"], sum(results.values()))*100
    prec = div_by_zero(results["true_positive"],  results["true_positive"] + results["false_positive"])*100
    rec  = div_by_zero(results["true_positive"],  results["true_positive"] + results["false_negative"])*100
    spec = div_by_zero(results["false_positive"], results["true_negative"] + results["false_positive"])*100
    # F    = div_by_zero(2, div_by_zero(1, prec) + div_by_zero(1, rec))*100
    m["Accuracy"] = acc
    m["Precision"] = prec
    m["Recall"] = rec
    m["Specificity"] = spec
    # m["F-measure"] = F
    return m

def metrics_average(D):
    aver = {}
    for i in D[0]:
        for x in D:
            aver[i] = (aver.get(i, 0) + x[i])
        aver[i] = aver[i]/len(D)
    return aver

def div_by_zero(x, y):
    if y != 0:
        return x*1.0/y
    else:
        return 0.0


# index = sys.argv[1]
name = sys.argv[1]

# q = open("train"+index+".csv", "r")
q = open(name+".csv", "r")
train_input = [a.strip().split(",") for a in q][1:]
# plus  = [a for a in train_input if a[-1] == "positive"]
# minus = [a for a in train_input if a[-1] == "negative"]
q.close()
# w = open("test"+index+".csv", "r")
# unknown = [a.strip().split(",") for a in w]
# w.close()

attrib_names = [
     "top-left-square",
   "top-middle-square",
    "top-right-square",
  "middle-left-square",
"middle-middle-square",
 "middle-right-square",
  "bottom-left-square",
"bottom-middle-square",
 "bottom-right-square",
               "class"
]

total = {
  "true_positive": 0,
 "false_negative": 0,
 "false_positive": 0,
  "true_negative": 0,
  "contradictory": 0,
}

K = 5
result = []
count = 0
split = k_fold_cross_validation(train_input, K, True)

for i in range(K):
    print "k = "+str(i+1)
    cv_res = {
      "true_positive": 0,
     "false_negative": 0,
     "false_positive": 0,
      "true_negative": 0,
      "contradictory": 0,
    }
    data = next(split)
    training = data[0]
    testing  = data[1]
    plus_k  = [a for a in train_input if a[-1] == "positive"]
    minus_k = [a for a in train_input if a[-1] == "negative"]

    for elem in testing:
        check_hypothesis(plus_k, minus_k, elem, 1)
        count += 1
        # print str(count)+"/"+str(len(train_input))
    result.append(metrics(cv_res))
    for c1 in ["true","false"]:
        for c2 in ["positive","negative"]:
            total[c1+"_"+c2] = total.get(c1+"_"+c2,0) + cv_res[c1+"_"+c2]

average = metrics_average(result)
print average
print total