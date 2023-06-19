# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2022/7/8, 2020/10/3, 2020/10/1
# @Author : Zhen Tian, Yupeng Hou, Zihan Lin
# @Email  : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn
import argparse
import nni

from recbole_cdr.quick_start import run_recbole_cdr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str,
                        default='BPR', help='name of models')
    parser.add_argument('--dataset', '-d', type=str,
                        default='ml-100k', help='name of datasets')
    parser.add_argument('--config_files', type=str,
                        default=None, help='config files')

    args, _ = parser.parse_known_args()

    tuner_params = nni.get_next_parameter()
    tuner_params['nni'] = True
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    rst = run_recbole_cdr(
        model=args.model,
        config_file_list=config_file_list,
        nni_para=tuner_params
    )

    test_result_source = rst['test_result_source']
    _ = {}
    for k, v in test_result_source.items():
        _[k + '_source'] = v
    test_result_source = _

    test_result_target = rst['test_result_target']
    for k, v in test_result_target.items():
        _[k + '_target'] = v
    test_result_target = _

    nni_result = {}
    nni_result['default'] = list(test_result_source.values())[0] + list(test_result_target.values())[0]
    nni_result.update(test_result_source)
    nni_result.update(test_result_target)
    nni.report_final_result(nni_result)
