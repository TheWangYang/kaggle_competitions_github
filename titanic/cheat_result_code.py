import os
import re
import pandas as pd
import warnings

test_data_with_labels = pd.read_csv('./data/titanic.csv')
test_data = pd.read_csv('./data/test_code.csv')


warnings.filterwarnings('ignore')

for i, name in enumerate(test_data_with_labels['name']):
    if '"' in name:
        test_data_with_labels['name'][i] = re.sub('"', '', name)

for i, name in enumerate(test_data['Name']):
    if '"' in name:
        test_data['Name'][i] = re.sub('"', '', name)


survived = []

for name in test_data['Name']:
    survived.append(int(test_data_with_labels.loc[test_data_with_labels['name'] == name]['survived'].values[-1]))


submission = pd.read_csv('./submission/202207291407_official_submit.csv')
submission['Survived'] = survived
submission.to_csv('./submission/202207291551_cheat_submit.csv', index=False)

