set -xe

cur_dir=$(dirname $(realpath $0))
project_dir=$(dirname $cur_dir)
python=python

exp_name=$1
exp_type=${2:-"mlp"}

pushd $project_dir
    log_dir=experiments/output/${exp_name}/
    run_dir=$log_dir/run_$(date +%Y%m%d_%H%M%S)
    mkdir -p $run_dir
    
    branch=$(git rev-parse --abbrev-ref HEAD)
    hash=$(git rev-parse HEAD)
    echo "Branch: $branch" >> $run_dir/git_info.txt
    echo "Hash: $hash" >> $run_dir/git_info.txt
    git diff > $run_dir/git_diff.txt
    script -q -c "${python} scripts/train_${exp_type}.py ${exp_name}" 2>&1 | \
        tee -a experiments/output/${exp_name}/train.log
popd