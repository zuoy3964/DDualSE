def get_feature_dict(dataset_dict, model_params, data, device):
    """
    Params:
        dataset: str
        model_name: str
        data: set
        device: torch.device
    Returns:
        feature_dict: dict
    """
    dataset = dataset_dict['dataset_name']
    items_cnt = dataset_dict['items_cnt']
    ebd_method = model_params['ebd_method']

    if dataset == "bookcrossing":
        history_seq = data[1].to(device)
        history_title = data[2].to(device)
        history_summary = data[3].to(device)
        history_author = data[4].to(device)
        history_category = data[5].to(device)
        history_year = data[6].to(device)

        candidate_id = data[7].to(device)
        candidate_title = data[8].to(device)
        candidate_summary = data[9].to(device)
        candidate_author = data[10].to(device)
        candidate_category = data[11].to(device)
        candidate_year = data[12].to(device)

        category_cnt = dataset_dict['category_cnt']
        year_cnt = dataset_dict['year_cnt']
        word_dim = dataset_dict['word_dim']

        if ebd_method == 'fine-grained':
            feature_dict = {
                f"text_{word_dim}": {
                    "history_title": history_title,
                    "candidate_title": candidate_title,
                    "history_summary": history_summary,
                    "candidate_summary": candidate_summary,
                    "history_author": history_author,
                    "candidate_author": candidate_author,
                    },
                f"categorial_{category_cnt}": {
                    "history_category": history_category,
                    "candidate_category": candidate_category,
                    },
                f"categorial_{year_cnt}": {
                    "history_year": history_year,
                    "candidate_year": candidate_year,
                    },
                }

        elif ebd_method == 'coarse-grained':
            feature_dict = {
                f"sequential_{items_cnt}": {
                    "history_seq": history_seq,
                    "candidate_seq": candidate_id,
                    }
                }

        elif ebd_method == "fine&coarse":
            feature_dict = {
                f"text_{word_dim}": {
                    "history_title": history_title,
                    "candidate_title": candidate_title,
                    "history_summary": history_summary,
                    "candidate_summary": candidate_summary,
                    "history_author": history_author,
                    "candidate_author": candidate_author,
                    },
                f"categorial_{category_cnt}": {
                    "history_category": history_category,
                    "candidate_category": candidate_category,
                    },
                f"categorial_{year_cnt}": {
                    "history_year": history_year,
                    "candidate_year": candidate_year,
                    },
                f"sequential_{items_cnt}": {
                    "history_seq": history_seq,
                    "candidate_seq": candidate_id,
                    }
                }

        else:
            raise ValueError("ebd_method must be in ('fine-grained','coarse-grained,'fine&coarse')")

    elif dataset == 'mindlarge':
        history_seq = data[12].to(device)
        history_category = data[1].to(device)
        history_subcateory = data[2].to(device)
        history_headline = data[3].to(device)
        history_abstract = data[4].to(device)

        candidate_id = data[5].to(device)
        candidate_category = data[6].to(device)
        candidate_subcategory = data[7].to(device)
        candidate_headline = data[8].to(device)
        candidate_abstract = data[9].to(device)

        category_cnt = dataset_dict['category_cnt']
        subcategory_cnt = dataset_dict['subcategory_cnt']
        word_dim = dataset_dict['word_dim']

        if ebd_method == 'fine-grained':
            feature_dict = {
                f"text_{word_dim}": {
                    "history_headline": history_headline,
                    "candidate_headline": candidate_headline,
                    "history_abstract": history_abstract,
                    "candidate_abstract": candidate_abstract,
                    },
                f"categorial_{category_cnt}": {
                    "history_category": history_category,
                    "candidate_category": candidate_category,
                    },
                f"categorial_{subcategory_cnt}": {
                    "history_subcateory": history_subcateory,
                    "candidate_subcategory": candidate_subcategory,
                    },
                }
        elif ebd_method == 'coarse-grained':
            feature_dict = {
                f"sequential_{items_cnt}": {
                    "history_seq": history_seq,
                    "candidate_seq": candidate_id,
                    }
                }

        elif ebd_method == "fine&coarse":
            feature_dict = {
                f"text_{word_dim}": {
                    "history_headline": history_headline,
                    "candidate_headline": candidate_headline,
                    "history_abstract": history_abstract,
                    "candidate_abstract": candidate_abstract,
                    },
                f"categorial_{category_cnt}": {
                    "history_category": history_category,
                    "candidate_category": candidate_category,
                    },
                f"categorial_{subcategory_cnt}": {
                    "history_subcateory": history_subcateory,
                    "candidate_subcategory": candidate_subcategory,
                    },
                f"sequential_{items_cnt}": {
                    "history_seq": history_seq,
                    "candidate_seq": candidate_id,
                    }
                }

        else:
            raise ValueError("ebd_method must be in ('fine-grained','coarse-grained,'fine&coarse')")

    elif dataset == "movielens1m":
        history_seq = data[1].to(device)
        history_title = data[2].to(device)
        history_category = data[3].to(device)
        history_year = data[4].to(device)
        candidate_id = data[5].to(device)
        candidate_title = data[6].to(device)
        candidate_category = data[7].to(device)
        candidate_year = data[8].to(device)

        word_dim = dataset_dict['word_dim']
        category_cnt = dataset_dict['category_cnt']
        year_cnt = dataset_dict['year_cnt']

        if ebd_method == 'fine-grained':

            feature_dict = {
                f"text_{word_dim}": {
                    "history_title": history_title,
                    "candidate_title": candidate_title,
                    },
                f"categorial_{category_cnt}": {
                    "history_category": history_category,
                    "candidate_category": candidate_category,
                    },
                f"categorial_{year_cnt}": {
                    "history_year": history_year,
                    "candidate_year": candidate_year,
                    },
                }

        elif ebd_method == 'coarse-grained':
            feature_dict = {
                f"sequential_{items_cnt}": {
                    "history_seq": history_seq,
                    "candidate_seq": candidate_id,
                    }
                }

        elif ebd_method == "fine&coarse":
            feature_dict = {
                f"text_{word_dim}": {
                    "history_title": history_title,
                    "candidate_title": candidate_title,
                    },
                f"categorial_{category_cnt}": {
                    "history_category": history_category,
                    "candidate_category": candidate_category,
                    },
                f"categorial_{year_cnt}": {
                    "history_year": history_year,
                    "candidate_year": candidate_year,
                    },
                f"sequential_{items_cnt}": {
                    "history_seq": history_seq,
                    "candidate_seq": candidate_id,
                    }
                }

        else:
            raise ValueError("ebd_method must be in ('fine-grained','coarse-grained','fine&coarse')")

    return feature_dict
