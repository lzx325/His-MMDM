set -e
module purge
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "conda not found"; exit 1;
fi

conda_env_name="His"
conda deactivate
conda activate "$conda_env_name"
export LD_LIBRARY_PATH="$HOME/anaconda3/envs/$conda_env_name/lib:$LD_LIBRARY_PATH"