from datasketch import MinHash


def get_minhash_signature(img_gray, num_perm=128, patch_size=8, stride=4, grid_resolution=3):
    m = MinHash(num_perm=num_perm)
    h, w = img_gray.shape

    for y in range(0, h - patch_size, stride):
        grid_y = int((y / h) * grid_resolution)
        for x in range(0, w - patch_size, stride):
            grid_x = int((x / w) * grid_resolution)
            patch = img_gray[y:y + patch_size, x:x + patch_size]
            token = f"{grid_x}_{grid_y}_{tuple(patch.flatten())}".encode('utf-8')
            m.update(token)

    return m


def compute_minhash_similarity(m1, m2):
    return m1.jaccard(m2)