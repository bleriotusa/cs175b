__author__ = 'Michael'

def prediction_accuracies(scheme: str):
    d = {'a': 0}
    l = [18, 29, 30, 41, 52, 60, 79, 80, 91, 100]
    predictor = 2
    for n in l:
        if n % 2 == 0:
            if predictor < 2:
                d['a'] += 1
                print(n, 'wrong for {}'.format(2))
                if predictor < 3:
                    predictor += 1
            else:
                if predictor < 3:
                    predictor += 1
        else:
            if predictor > 1:
                d['a'] += 1
                print(n, 'wrong for {}'.format(2))
                if predictor > 0:
                    predictor -= 1
            else:
                if predictor < 3:
                    predictor += 1
        if n % 10 == 0:
            if predictor < 2:
                d['a'] += 1
                print(n, 'wrong for {}'.format(10))
                if predictor < 3:
                    predictor += 1
            else:
                if predictor < 3:
                    predictor += 1
        else:
            if predictor > 1:
                d['a'] += 1
                print(n, 'wrong for {}'.format(10))
                if predictor > 0:
                    predictor -= 1
            else:
                if predictor < 3:
                    predictor += 1

    print(d['a'])

prediction_accuracies('a')