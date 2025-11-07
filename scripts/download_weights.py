from __future__ import print_function
import os
from builtins import input
import shutil
from urllib.request import urlopen

curr_folder = os.path.basename(os.path.normpath(os.getcwd()))

weights_filename = 'pytorch_model.bin'
weights_folder = 'model'
weights_path = '{}/{}'.format(weights_folder, weights_filename)
if curr_folder == 'scripts':
    weights_path = '../' + weights_path
weights_download_link = 'https://www.dropbox.com/s/q8lax9ary32c7t9/pytorch_model.bin?dl=1'


MB_FACTOR = float(1<<20)

def prompt():
    while True:
        valid = {
            'y': True,
            'ye': True,
            'yes': True,
            'n': False,
            'no': False,
        }
        choice = input().lower()
        if choice in valid:
            return valid[choice]
        else:
            print('Please respond with \'y\' or \'n\' (or \'yes\' or \'no\')')

download = True
if os.path.exists(weights_path):
    print('Weight file already exists at {}. Would you like to redownload it anyway? [y/n]'.format(weights_path))
    download = prompt()
    already_exists = True
else:
    already_exists = False

if download:
    print('About to download the pretrained weights file from {}'.format(weights_download_link))
    if already_exists == False:
        print('The size of the file is roughly 85MB. Continue? [y/n]')
    else:
        os.unlink(weights_path)

    if already_exists or prompt():
        print('Downloading...')

        #urllib.urlretrieve(weights_download_link, weights_path)
        #with open(weights_path,'wb') as f:
        #    f.write(requests.get(weights_download_link).content)

        abs_weights_path = os.path.abspath(weights_path)
        weights_dir = os.path.dirname(abs_weights_path)
        if weights_dir and not os.path.exists(weights_dir):
            os.makedirs(weights_dir)

        with urlopen(weights_download_link) as response:
            content_type = response.info().get_content_type()
            if content_type and content_type.startswith('text'):
                raise ValueError(
                    'Download failed. The server returned unexpected content type: {}.\n'
                    'The download link might be invalid or require additional authentication.'
                    .format(content_type)
                )

            with open(abs_weights_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)

        if os.path.getsize(weights_path) / MB_FACTOR < 80:
            raise ValueError("Download finished, but the resulting file is too small! " +
                             "It\'s only {} bytes.".format(os.path.getsize(weights_path)))
        print('Downloaded weights to {}'.format(weights_path))
else:
    print('Exiting.')
