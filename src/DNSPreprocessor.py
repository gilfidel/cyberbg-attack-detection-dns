def update_rel_frame_time(org_frame_time, duration):
    return round(org_frame_time - duration, 7)


def replace_src_with_dst(col_name):
    if 'src' in col_name:
        col_name = col_name.replace('src', 'dst')
    else:
        if 'dst' in col_name:
            col_name = col_name.replace('dst', 'src')

    return col_name


def preprocess_client_dsp_data(csv_file_path):
    import pandas as pd

    client_data = pd.read_csv(csv_file_path)

    client_data.columns = [col.replace('.', '_') for col in client_data.columns]

    client_data_dts = {k: v for k, v in client_data.groupby('ip_src')}

    client_data_keys = client_data_dts.keys()

    result = filter(lambda s: '~' not in s, client_data_dts.keys())

    client_data_keys = list(result)

    src = client_data['ip_src'][0]

    dst_result = filter(lambda s: not (s == src), client_data_keys)

    dst_list = list(dst_result)

    dst = dst_list[0]

    ds_responses = client_data_dts[dst]

    ds_responses['frame_time_relative'] = ds_responses.apply(
        lambda row: update_rel_frame_time(row.frame_time_relative, row.dns_time), axis=1)

    ds_responses.rename(columns=replace_src_with_dst, inplace=True)

    ds_responses.rename({'dns_flags': 'dns_response_flags'}, axis=1, inplace=True)

    joined_fields = ['frame_time_relative',
                     'ip_src',
                     'ip_dst',
                     'udp_srcport',
                     'udp_dstport',
                     'tcp_dstport',
                     'tcp_srcport',
                     'dns_qry_type',
                     'dns_qry_name',
                     ]

    responses_selected_fields = [
        'dns_resp_type',
        'dns_resp_name',
        'dns_resp_ttl',
        'dns_a',
        'dns_aaaa',
        'dns_cname',
        'dns_response_flags',
        'dns_flags_response',
        'dns_flags_rcode',
        'dns_count_queries',
        'dns_count_answers',
        'dns_count_auth_rr',
        'dns_count_add_rr',
        'dns_soa_mname',
        'dns_srv_name',
        'dns_time'
    ]

    queries_selected_fields = [
        'dns_flags'
    ]

    ds_queries = client_data_dts[src][joined_fields + queries_selected_fields]

    ds_queries.rename({'dns_flags': 'dns_query_flags'}, axis=1, inplace=True)

    ds_responses = ds_responses[joined_fields + responses_selected_fields]

    merged_ds = pd.merge(ds_queries, ds_responses, on=joined_fields)

    merged_ds = merged_ds.drop(
        ['ip_src',
         'ip_dst',
         'udp_srcport',
         'udp_dstport',
         'tcp_dstport',
         'tcp_srcport'], 1)

    return merged_ds
