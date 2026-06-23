def get_recommendations(data):

    recommendations = []

    if data["Contract"] == "Month-to-month":
        recommendations.append(
            "Offer discounted annual contract."
        )

    if data["tenure"] < 12:
        recommendations.append(
            "Provide onboarding incentives."
        )

    if data["MonthlyCharges"] > 80:
        recommendations.append(
            "Offer loyalty discount."
        )

    if data["PaymentMethod"] == "Electronic check":
        recommendations.append(
            "Promote automatic bank payments."
        )

    return recommendations