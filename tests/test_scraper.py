from src.scraper import norm_url, reg_domain, origin_of


def test_norm_url():
    assert norm_url("ibm.com") == "https://ibm.com"
    assert norm_url("http://oracle.com/") == "http://oracle.com"
    assert norm_url("https://www.google.com") == "https://www.google.com"


def test_reg_domain():
    assert reg_domain("www.ibm.com") == "ibm.com"
    assert reg_domain("sub.domain.co.uk") == "domain.co.uk"
    assert reg_domain("portal.app.startup.io") == "startup.io"


def test_origin_of():
    assert (
        origin_of("https://www.ibm.com/trust/privacy-policy") == "https://www.ibm.com"
    )
    assert origin_of("http://sub.test.com/about?user=123") == "http://sub.test.com"
