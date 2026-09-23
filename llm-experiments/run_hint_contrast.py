"""Prepare alternative incorrect hints, measure saved policies, or reconstruct their contrast."""
import argparse
from src.hint_contrast import prepare_control, measure_control, analyze_contrast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--study',required=True)
    prepare.add_argument('--output',required=True)
    for command in ('measure','analyze'):
        p=commands.add_parser(command)
        p.add_argument('--study',required=True)
        p.add_argument('--control',required=True)
        p.add_argument('--run-id',required=True)
        if command=='measure':
            p.add_argument('--measurement-id',default='alternative_wrong_n512_batch64')
            p.add_argument('--question-batch-size',type=int,default=64)
        else:
            p.add_argument('--original-id',default='n512_batch64')
            p.add_argument('--alternative-id',default='alternative_wrong_n512_batch64')
            p.add_argument('--output',required=True)
    a=parser.parse_args()
    if a.command=='prepare': result=prepare_control(a.study,a.output)
    elif a.command=='measure': result=measure_control(a.study,a.control,a.run_id,a.measurement_id,a.question_batch_size)
    else: result=analyze_contrast(a.study,a.control,a.run_id,a.output,a.original_id,a.alternative_id)
    print(result)

if __name__=='__main__':
    main()
