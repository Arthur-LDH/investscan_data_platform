FROM apache/nifi:latest

USER root

# Détection et installation du gestionnaire de paquets approprié
RUN if command -v apt-get >/dev/null 2>&1; then \
        apt-get update && apt-get install -y wget && apt-get clean; \
    elif command -v yum >/dev/null 2>&1; then \
        yum install -y wget && yum clean all; \
    elif command -v apk >/dev/null 2>&1; then \
        apk update && apk add --no-cache wget; \
    elif command -v dnf >/dev/null 2>&1; then \
        dnf install -y wget && dnf clean all; \
    else \
        echo "Aucun gestionnaire de paquets reconnu trouvé"; \
        exit 1; \
    fi

# Installer les NAR supplémentaires pour S3
RUN wget https://repo1.maven.org/maven2/org/apache/nifi/nifi-aws-nar/1.19.1/nifi-aws-nar-1.19.1.nar -P /opt/nifi/nifi-current/lib/ && \
    wget https://repo1.maven.org/maven2/org/apache/nifi/nifi-aws-service-api-nar/1.19.1/nifi-aws-service-api-nar-1.19.1.nar -P /opt/nifi/nifi-current/lib/

# Vérifier que les fichiers ont bien été téléchargés
RUN ls -la /opt/nifi/nifi-current/lib/nifi-aws-*.nar

# Définir les permissions appropriées
RUN chown -R nifi:nifi /opt/nifi/nifi-current/lib/nifi-aws-*.nar && \
    chmod 644 /opt/nifi/nifi-current/lib/nifi-aws-*.nar

# Revenir à l'utilisateur NiFi
USER nifi