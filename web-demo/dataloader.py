import numpy as np
import pickle


class DataLoader(object):
    compositions = {
        'seen': [
            (30, 10, 60),
            (60, 30, 10),
            (10, 40, 50),
            (30, 60, 10),
            (10, 50, 40),
            (30, 20, 50),
            (50, 10, 40),
            (40, 50, 10),
            (50, 30, 20)],
        'unseen': [
                (50, 40, 10),
                (70, 10, 20),
                (20, 70, 10),
                (10, 20, 70),
                (70, 20, 10),
                (10, 70, 20),
                (20, 10, 70),
                (10, 60, 30),
                (40, 10, 50)
            ]
    }

    def __init__(self, path):
        with open(path, 'rb') as f:
            self.results = pickle.load(f)

    def get_options_shape(self):
        return [
            {'label': 'Two peak hours per day', 'value': 'waves'},
            {'label': 'Roughly stable', 'value': 'steps'}
        ]

    def get_options_multiplier(self, shape):
        if shape == 'waves':
            return 1, 3
        else:
            return 1, 1

    def get_options_composition(self, shape, multiplier):
        if shape == 'steps':
            return self.compositions['seen']
        else:
            return self.compositions['seen'] + self.compositions['unseen']

    def get_datasets(self):
        return list(self.results.keys())

    def get_learning_traffic(self):
        calls = self.results['composePost_uploadMedia_readUserTimeline-waves_waves-seen_compositions-1x']['nginx-thrift']['cpu']['calls']
        return {
            'ALL': calls[0][:9*60] + calls[1][:9*60] + calls[2][:9*60],
            '/composePost': calls[0][:9*60],
            '/uploadMedia': calls[1][:9*60],
            '/readTimeline': calls[2][:9*60]
        }

    def get_query_traffic(self, shape, multiplier, composition):
        composition = tuple(map(int, composition.split('_')))
        assert composition in self.compositions['unseen'] or composition in self.compositions['seen']
        composition_name = ('unseen' if composition in self.compositions['unseen'] else 'seen')
        dbname = 'composePost_uploadMedia_readUserTimeline-waves_%s-%s_compositions-%dx' % (
            shape, composition_name, int(multiplier)
        )
        calls = self.results[dbname]['nginx-thrift']['cpu']['calls']
        offset = 9*60
        index = self.compositions[composition_name].index(composition)
        outs = {
            '/composePost': calls[0][offset + index * 60:offset + index * 60 + 60],
            '/uploadMedia': calls[1][offset + index * 60:offset + index * 60 + 60],
            '/readTimeline': calls[2][offset + index * 60:offset + index * 60 + 60]
        }
        outs['ALL'] = outs['/composePost'] + outs['/uploadMedia'] + outs['/readTimeline']
        return outs

    def get_component2metrics(self, shape, multiplier, composition):
        composition = tuple(map(int, composition.split('_')))
        assert composition in self.compositions['unseen'] or composition in self.compositions['seen']
        composition_name = ('unseen' if composition in self.compositions['unseen'] else 'seen')
        dbname = 'composePost_uploadMedia_readUserTimeline-waves_%s-%s_compositions-%dx' % (
            shape, composition_name, int(multiplier)
        )

        keys = list(set(key.replace('-pvc', '') for key in self.results[dbname].keys()))
        outs = {}
        for key in keys:
            icon_path = './assets/component_container.png'
            if 'mongodb' in key:
                icon_path = './assets/component_mongodb.png'
            elif 'nginx' in key or 'frontend' in key:
                icon_path = './assets/component_nginx.png'
            names = {'nginx-thrift': 'NGINX Thrift',
                     'media-frontend': 'Media Frontend',
                     'media-mongodb': 'Media MongoDB',
                     'post-storage-service': 'Post Storage Service',
                     'post-storage-mongodb': 'Post Storage MongoDB',
                     'compose-post-service': 'Compose Post Service',
                     'user-timeline-service': 'User Timeline Service',
                     'user-timeline-mongodb': 'User Timeline MongoDB'}
            metric2scale = {}
            metric2utilization = {}
            composition_name = ('unseen' if composition in self.compositions['unseen'] else 'seen')
            index = self.compositions[composition_name].index(composition)
            for metric in ['cpu', 'memory', 'write-iops', 'write-tp', 'usage']:
                if metric in self.results[dbname][key]:
                    metric2scale[metric] = [
                        self.results[dbname][key][metric]['scale_groundtruth'][index],
                        self.results[dbname][key][metric]['scale_bl-resrc'][index],
                        self.results[dbname][key][metric]['scale_bl-api'][index],
                        self.results[dbname][key][metric]['scale_bl-trace'][index],
                        self.results[dbname][key][metric]['scale_ours'][index]
                    ]
                    metric2utilization[metric] = [
                        np.asarray(self.results[dbname][key][metric]['measurement'][2 * 60:9 * 60] + self.results[dbname][key][metric]['measurement'][9 * 60 + index * 60:9 * 60 + index * 60 + 60]),
                        np.asarray(self.results[dbname][key][metric]['prediction_bl-resrc'][index * 60: index * 60 + 60]),
                        np.asarray(self.results[dbname][key][metric]['prediction_bl-api'][index * 60: index * 60 + 60]),
                        np.asarray(self.results[dbname][key][metric]['prediction_bl-trace'][index * 60: index * 60 + 60]),
                        np.asarray(self.results[dbname][key][metric]['prediction_ours'][index * 60: index * 60 + 60])
                    ]
                elif key+'-pvc' in self.results[dbname]:
                    metric2scale[metric] = [
                        self.results[dbname][key+'-pvc'][metric]['scale_groundtruth'][index],
                        self.results[dbname][key+'-pvc'][metric]['scale_bl-resrc'][index],
                        self.results[dbname][key+'-pvc'][metric]['scale_bl-api'][index],
                        self.results[dbname][key+'-pvc'][metric]['scale_bl-trace'][index],
                        self.results[dbname][key+'-pvc'][metric]['scale_ours'][index]
                    ]
                    metric2utilization[metric] = [
                        np.asarray(self.results[dbname][key + '-pvc'][metric]['measurement'][2 * 60:9 * 60] + self.results[dbname][key + '-pvc'][metric]['measurement'][9 * 60 + index * 60:9 * 60 + index * 60 + 60]),
                        np.asarray(self.results[dbname][key + '-pvc'][metric]['prediction_bl-resrc'][index * 60: index * 60 + 60]),
                        np.asarray(self.results[dbname][key + '-pvc'][metric]['prediction_bl-api'][index * 60: index * 60 + 60]),
                        np.asarray(self.results[dbname][key + '-pvc'][metric]['prediction_bl-trace'][index * 60: index * 60 + 60]),
                        np.asarray(self.results[dbname][key + '-pvc'][metric]['prediction_ours'][index * 60: index * 60 + 60])
                    ]
                else:
                    metric2scale[metric] = [0., 0., 0., 0., 0.]
                if metric in metric2utilization and (metric == 'memory' or metric == 'usage'):
                    gt_offset = metric2utilization[metric][0][7*60-1]
                    metric2utilization[metric][1] = metric2utilization[metric][1] - metric2utilization[metric][1][0] + gt_offset
                    metric2utilization[metric][2] = metric2utilization[metric][2] - metric2utilization[metric][2][0] + gt_offset
                    metric2utilization[metric][3] = metric2utilization[metric][3] - metric2utilization[metric][3][0] + gt_offset
                    metric2utilization[metric][4] = metric2utilization[metric][4] - metric2utilization[metric][4][0] + gt_offset
                    metric2utilization[metric][0][7*60:] = metric2utilization[metric][0][7*60:] - metric2utilization[metric][0][7*60:][0] + gt_offset
                    metric2scale[metric] = [
                        max(metric2utilization[metric][0][7*60:]) / max(metric2utilization[metric][0][:7*60]),
                        max(metric2utilization[metric][1]) / max(metric2utilization[metric][0][:7*60]),
                        max(metric2utilization[metric][2]) / max(metric2utilization[metric][0][:7*60]),
                        max(metric2utilization[metric][3]) / max(metric2utilization[metric][0][:7*60]),
                        max(metric2utilization[metric][4]) / max(metric2utilization[metric][0][:7*60])
                    ]

            outs[key] = {
                'id': key,
                'name': names[key],
                'icon_path': icon_path,
                'metrics': ['cpu', 'memory', 'write-iops', 'write-tp', 'usage'],
                'unit': {'cpu': 'millicores', 'memory': 'MB', 'write-iops': 'IOps', 'write-tp': 'KB', 'usage': 'MB'},
                'scale': metric2scale,
                'utilization': metric2utilization
            }
        return outs


if __name__ == '__main__':
    dl = DataLoader('assets/results.pkl')
    print(dl.get_learning_traffic())
