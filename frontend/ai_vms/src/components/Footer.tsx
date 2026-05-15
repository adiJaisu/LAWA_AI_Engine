import "../styles/footer.css";

const Footer = () => {
  return (
    <footer className="footer">
      <div className="footer-content">
        <div className="footer-left">
          <p>
            Copyright © 2026 <b>HCLTech</b> and its related entities. All Rights
            Reserved.
          </p>
        </div>

        <div className="footer-right">
          <div className="version-info">
            <span>v0.1.1</span>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
