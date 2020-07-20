import os
import tensorflow as tf

######################
# parametres
######################
base_dir = "./data/invariant_beer_release"
glove_path = "./embeddings/glove.6B.100d.txt"

aspect = 0
rationale_length = 10

num_restart = 5
diff_lambda_list = [0.5, 1., 2., 5., 10., 20., 50., 100.]

gpu = "0"

######################
# output file
######################
output_dir = "./beer_model_selection/aspect_%d" % (aspect)
if not tf.gfile.Exists(output_dir):
    tf.gfile.MakeDirs(output_dir)

output_file = os.path.join(output_dir,
                           "length_%d_results.txt" % rationale_length)

headline = "rationale_length\tdiff_lambda\tbest_dev_epoch\tdev_inv_acc\tdev_enb_acc\tcors_test_sparsity\tcors_test_precision\tcors_test_recall\tcors_test_f1\n"

f = open(output_file, "wt")
f.write(headline)
f.close()

for diff_lambda in diff_lambda_list:
    f = open(output_file, "a")
    f.write("-----------------------------------------------------------\n")
    f.close()
    for i in range(num_restart):
        command = "python run_beer.py --base_dir %s --glove_path %s --output_file %s --aspect %d --rationale_length %d --diff_lambda %f --gpu %s" % (
            base_dir, glove_path, output_file, aspect, rationale_length,
            diff_lambda, gpu)
        os.system(command)
