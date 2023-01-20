import clime

if __name__ == '__main__':
    opts = {
        'dataset':             'moons',
        'class samples':       [25, 75],
        'dataset rebalancing': 'none',
        'model':               'SVM',
        'model balancer':      'none',
        'explainer':           'normal',
        'query point':         15,
        'evaluation':          'normal',
    }
    p = clime.pipeline.construct(opts)
    print(p.run())
