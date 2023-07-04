import logging
from params import *
from twilio.rest import Client

# logger = logging.getLogger(__name__)


@st.cache_data
def get_ice_servers():
    """Use Twilio's TURN server because Streamlit Community Cloud has changed
    its infrastructure and WebRTC connection cannot be established without TURN server now.
    """
    # Ref: https://www.twilio.com/docs/stun-turn/api
    try:
        account_sid = TWILIO_ACCOUNT_SID
        auth_token = TWILIO_AUTH_TOKEN
    except KeyError:
        # logger.warning(
        #     "Twilio credentials are not set. Fallback to a free STUN server from Google."  # noqa: E501
        # )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)

    token = client.tokens.create()

    return token.ice_servers
