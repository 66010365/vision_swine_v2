export default function PageHeader({
  title,
  subtitle,
  action,
}) {
  return (
    <header className="page-header">
      <div className="page-header-content">
        <h1 className="page-header-title">{title}</h1>

        {subtitle && (
          <p className="page-header-subtitle">
            {subtitle}
          </p>
        )}
      </div>

      {action && (
        <div className="page-header-action">
          {action}
        </div>
      )}
    </header>
  );
}