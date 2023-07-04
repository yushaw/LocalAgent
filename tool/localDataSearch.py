def localDataSearch(query: str, database = Data_chat_bot) -> str:
    """
    search the local database for the given query.
    
    :param query: the subject you want to search
    :return: The answer to the query
    """
    
    return Data_chat_bot.query(query)