from experimentbuilder import ExperimentBuilder


experiment = ExperimentBuilder()\
    .with_noise({'snp': [0.2, 0.4, 0.6]})\
    .with_noise({'rot': [15, 30, 60]})\
    .build()
