package org.snu.cse.cap.translator.structure.device.connection;

public class SSLKeyInfo {
	private String caPublicKey;
	private String publicKey;
	private String privateKey;

	public SSLKeyInfo(String caPublicKey, String publicKey, String privateKey) {
		this.caPublicKey = caPublicKey;
		this.publicKey = publicKey;
		this.privateKey = privateKey;
	}

	public String getCaPublicKey() {
		return caPublicKey;
	}

	public String getPublicKey() {
		return publicKey;
	}

	public String getPrivateKey() {
		return privateKey;
	}

	public boolean equals(Object obj) {
		return (caPublicKey.equals(((SSLKeyInfo) obj).caPublicKey) && publicKey.equals(((SSLKeyInfo) obj).publicKey)
				&& privateKey.equals(((SSLKeyInfo) obj).privateKey));
	}
}
