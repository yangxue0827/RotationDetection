# -*- coding: utf-8 -*-
import os
import requests
import hashlib
import zipfile
from tqdm import tqdm


def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.
    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.
    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    sha1_file = sha1.hexdigest()
    l = min(len(sha1_file), len(sha1_hash))
    return sha1.hexdigest()[0:l] == sha1_hash[0:l]


def download(url, path=None, overwrite=False, sha1_hash=None):
    """Download an given URL
    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split('/')[-1]
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path

    if overwrite or not os.path.exists(fname) or (sha1_hash and not check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        print('Downloading %s from %s...'%(fname, url))
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s"%url)
        total_length = r.headers.get('content-length')
        with open(fname, 'wb') as f:
            if total_length is None: # no content length header
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(r.iter_content(chunk_size=1024),
                                  total=int(total_length / 1024. + 0.5),
                                  unit='KB', unit_scale=False, dynamic_ncols=True):
                    f.write(chunk)

        if sha1_hash and not check_sha1(fname, sha1_hash):
            raise UserWarning('File {} is downloaded but the content hash does not match. ' \
                              'The repo may be outdated or download may be incomplete. ' \
                              'If the "repo_url" is overridden, consider switching to ' \
                              'the default repo.'.format(fname))

    return fname


def download_mxnet_weights(name, tag=None, root='../mxnet_weights'):
    # https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/model_store.py
    _model_sha1 = {name: checksum for checksum, name in [
        ('2d9d980c990442f826f20781ed039851e78dabe3', 'resnet18_v1b'),
        ('8e16b84814e84f64d897854003f049872991eaa6', 'resnet34_v1b'),
        ('0ecdba34691be172036ddf244ff1b2eade75ffde', 'resnet50_v1b'),
        ('a455932aa95cb7dcfa05fd040b9b5a5660733c39', 'resnet101_v1b'),
        ('a5a61ee1ce5ab7c09720775b223360f3c60e211d', 'resnet152_v1b'),
        ('2a4e070854db538595cc7ee02e1a914bdd49ca02', 'resnet50_v1c'),
        ('064858f23f9878bfbbe378a88ccb25d612b149a1', 'resnet101_v1c'),
        ('75babab699e1c93f5da3c1ce4fd0092d1075f9a0', 'resnet152_v1c'),
        ('117a384ecf61490eb31ea147eb0e61e6d2b8a449', 'resnet50_v1d'),
        ('1b2b825feff86b0354642a4ab59f9b6e35e47338', 'resnet101_v1d'),
        ('cddbc86ff24a5544f57242ded0acb14ef1fbd437', 'resnet152_v1d'),
        ('25a187fa281ddc98afbcd0cc0f0646885b874b80', 'resnet50_v1s'),
        ('bd93a83c05f709a803b1221aeff0b028e6eebb03', 'resnet101_v1s'),
        ('cf74621d988ad06c6c6aa44f5597e5b600a966cc', 'resnet152_v1s'),
        ('11c50114a0483e27e74dc4236904254ef05b634b', 'SE_ResNext101_64x4d'),
        ('364590740605b6a2b95f5bb77436d781a817436f', 'resnest26'),
    ]}

    apache_repo_url = 'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/'
    _url_format = '{repo_url}gluon/models/{file_name}.zip'

    def short_hash(name):
        if name not in _model_sha1:
            raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
        return _model_sha1[name][:8]

    use_tag = isinstance(tag, str)
    if use_tag:
        file_name = '{name}-{short_hash}'.format(name=name,
                                                 short_hash=tag)
    else:
        file_name = '{name}-{short_hash}'.format(name=name,
                                                 short_hash=short_hash(name))
    root = os.path.expanduser(root)
    file_path = os.path.join(root, file_name + '.params')
    if use_tag:
        sha1_hash = tag
    else:
        sha1_hash = _model_sha1[name]
    if os.path.exists(file_path):
        if check_sha1(file_path, sha1_hash):
            return file_path
        else:
            print('Mismatch in the content of model file detected. Downloading again.')
    else:
        print('Model file is not found. Downloading.')

    if not os.path.exists(root):
        os.makedirs(root)

    zip_file_path = os.path.join(root, file_name + '.zip')
    repo_url = os.environ.get('MXNET_GLUON_REPO', apache_repo_url)
    if repo_url[-1] != '/':
        repo_url = repo_url + '/'
    download(_url_format.format(repo_url=repo_url, file_name=file_name),
             path=zip_file_path,
             overwrite=True)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(root)
    os.remove(zip_file_path)

    if check_sha1(file_path, sha1_hash):
        return file_path
    else:
        raise ValueError('Downloaded file has different hash. Please try again.')


if __name__ == '__main__':
    # download_mxnet_weights('resnet%d_v%dd' % (50, 1), tag=True)
    download_mxnet_weights('resnest26', tag=True)